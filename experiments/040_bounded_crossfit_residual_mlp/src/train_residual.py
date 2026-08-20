"""평가연도와 분리된 bounded residual MLP cross-fitting 실행기."""
from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import psutil
from torch.utils.data import DataLoader, TensorDataset

from build_dataset import build_frame, build_test_frame
from contract import CAPS, ID, MODEL_FEATURES, NUMERIC, ONEHOT, PREDICTION_FEATURES, SNAPSHOTS, atomic_write_bytes, experiment_root, load_json, sha256
from evaluate import evaluate
from inspect_assets import inspect
from model import ResidualMLP, final_probability, make_preprocessor, numpy_probability


def atomic_json(path: Path, value: object) -> None:
    atomic_write_bytes(path, json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8"))


def atomic_torch(path: Path, value: object) -> None:
    buffer = io.BytesIO(); torch.save(value, buffer); atomic_write_bytes(path, buffer.getvalue())


def atomic_joblib(path: Path, value: object) -> None:
    buffer = io.BytesIO(); joblib.dump(value, buffer); atomic_write_bytes(path, buffer.getvalue())


def set_seed(seed: int) -> None:
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def row_digest(frame: pd.DataFrame) -> str:
    digest=hashlib.sha256()
    for value in frame[ID].astype(str): digest.update(value.encode("utf-8")); digest.update(b"\0")
    return digest.hexdigest()


def rss_mib(config: dict | None = None) -> float:
    value=psutil.Process().memory_info().rss/1024/1024
    if config is not None: config["_peak_rss_mib"]=max(float(config.get("_peak_rss_mib",0)),value)
    return value


def estimate_memory(frame: pd.DataFrame, config: dict, reserve_gib: float,
                    enforce: bool = True) -> dict:
    categories=sum(int(frame[c].nunique(dropna=True))+1 for c in ONEHOT)
    input_dim=len(NUMERIC)*2+1+categories+len(PREDICTION_FEATURES)
    largest_train=int(frame.fold.isin([2022,2023]).sum()); largest_predict=int(frame.fold.eq(2024).sum())
    transformed=(largest_train+largest_predict)*input_dim*(8+4)
    frame_bytes=int(frame.memory_usage(index=True,deep=True).sum())
    estimated=int((transformed+frame_bytes)*1.35+512*1024*1024)
    available=int(psutil.virtual_memory().available); reserve=int(reserve_gib*1024**3)
    result={"estimated_input_dim":input_dim,"largest_train_rows":largest_train,"largest_predict_rows":largest_predict,
            "frame_gib":frame_bytes/1024**3,"estimated_peak_gib":estimated/1024**3,
            "available_gib":available/1024**3,"required_reserve_gib":reserve_gib}
    print("메모리 사전 추정: "+json.dumps(result,ensure_ascii=False),flush=True)
    if enforce and estimated+reserve>available:
        raise MemoryError(f"안전 RAM 부족: 예상 peak={estimated/1024**3:.2f} GiB, 현재 available={available/1024**3:.2f} GiB, reserve={reserve_gib:.2f} GiB")
    return result


def predict_raw(model: ResidualMLP, values: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval(); chunks=[]
    with torch.no_grad():
        for start in range(0,len(values),batch_size):
            x=torch.as_tensor(values[start:start+batch_size],dtype=torch.float32,device=device)
            chunks.append(model(x).cpu().numpy())
    return np.concatenate(chunks)


def fit_snapshots(train: pd.DataFrame, predict: pd.DataFrame, seed: int, stage: str,
                  checkpoint_dir: Path, device: torch.device, config: dict) -> np.ndarray:
    completed=checkpoint_dir/stage/f"seed_{seed}"/"complete.json"
    raw_path=completed.parent/"raw_prediction.npy"
    train_digest=row_digest(train); predict_digest=row_digest(predict)
    signature={key:config[key] for key in ["batch_size","learning_rate","weight_decay","epochs","train_cap"]}
    signature.update({key:config.get(key) for key in ["_config_sha256","_input_asset_sha256"]})
    if completed.is_file() and raw_path.is_file():
        marker=json.loads(completed.read_text(encoding="utf-8"))
        saved=np.load(raw_path,allow_pickle=False)
        if (marker.get("status")=="complete" and marker.get("rows")==len(predict)
                and marker.get("train_row_id_sha256")==train_digest and marker.get("predict_row_id_sha256")==predict_digest
                and marker.get("training_signature")==signature and saved.shape==(len(predict),) and np.isfinite(saved).all()):
            return saved
    started=time.perf_counter(); set_seed(seed); completed.parent.mkdir(parents=True,exist_ok=True)
    preprocess_started=time.perf_counter(); pre_path=completed.parent/"preprocessor.joblib"; pre_marker_path=completed.parent/"preprocessor_complete.json"
    pre=None; x_train=None
    if pre_path.is_file() and pre_marker_path.is_file():
        try:
            pre_marker=json.loads(pre_marker_path.read_text(encoding="utf-8"))
            if pre_marker.get("train_row_id_sha256")==train_digest and pre_marker.get("training_signature")==signature:
                pre=joblib.load(pre_path)
        except Exception as exc: print(f"손상 preprocessing checkpoint 무시: {type(exc).__name__}: {exc}",flush=True)
    if pre is None:
        pre=make_preprocessor(); x_train=np.asarray(pre.fit_transform(train[MODEL_FEATURES]),dtype=np.float32); atomic_joblib(pre_path,pre)
        atomic_json(pre_marker_path,{"status":"complete","train_row_id_sha256":train_digest,"training_signature":signature})
    if x_train is None: x_train=np.asarray(pre.transform(train[MODEL_FEATURES]),dtype=np.float32)
    x_predict=np.asarray(pre.transform(predict[MODEL_FEATURES]),dtype=np.float32)
    print(f"{stage} seed={seed} preprocessing_seconds={time.perf_counter()-preprocess_started:.1f} input_dim={x_train.shape[1]} rss_mib={rss_mib(config):.1f}",flush=True)
    y=torch.as_tensor(train.target.to_numpy(np.float32)); base=torch.as_tensor(train.p_915.to_numpy(np.float32))
    dataset=TensorDataset(torch.from_numpy(x_train),y,base)
    generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(dataset,batch_size=int(config["batch_size"]),shuffle=True,generator=generator,num_workers=0,pin_memory=device.type=="cuda")
    model=ResidualMLP(x_train.shape[1]).to(device)
    zero=predict_raw(model,x_predict[:min(1024,len(x_predict))],device,int(config["batch_size"]))
    if not np.array_equal(zero,np.zeros_like(zero)):
        raise AssertionError("마지막 residual layer 0 초기화 실패")
    optimizer=torch.optim.Adam(model.parameters(),lr=float(config["learning_rate"]),weight_decay=float(config["weight_decay"]))
    snapshot_predictions=[]; start_epoch=1
    for candidate_epoch in range(int(config["epochs"]),0,-1):
        candidate=completed.parent/f"epoch_{candidate_epoch}.pt"
        if not candidate.is_file(): continue
        try:
            state=torch.load(candidate,map_location=device,weights_only=False)
            if (state.get("seed")!=seed or state.get("input_dim")!=x_train.shape[1]
                    or state.get("train_row_id_sha256")!=train_digest or state.get("predict_row_id_sha256")!=predict_digest
                    or state.get("training_signature")!=signature): continue
            model.load_state_dict(state["state_dict"]); optimizer.load_state_dict(state["optimizer"]); generator.set_state(state["generator_state"]); start_epoch=candidate_epoch+1
            if "torch_rng_state" in state: torch.set_rng_state(state["torch_rng_state"].cpu())
            if "numpy_rng_state" in state: np.random.set_state(state["numpy_rng_state"])
            for epoch in SNAPSHOTS:
                snapshot_file=completed.parent/f"epoch_{epoch}.pt"
                if epoch<=candidate_epoch and snapshot_file.is_file():
                    saved=torch.load(snapshot_file,map_location="cpu",weights_only=False).get("prediction")
                    if saved is not None and np.asarray(saved).shape==(len(predict),): snapshot_predictions.append(np.asarray(saved))
            print(f"{stage} seed={seed}: epoch {candidate_epoch} checkpoint에서 resume",flush=True); break
        except Exception as exc:
            print(f"손상 checkpoint 무시: {candidate} ({type(exc).__name__}: {exc})",flush=True)
    for epoch in range(start_epoch,int(config["epochs"])+1):
        epoch_started=time.perf_counter(); model.train(); total=0.0
        for xb,yb,bb in loader:
            xb,yb,bb=xb.to(device),yb.to(device),bb.to(device); optimizer.zero_grad(set_to_none=True)
            raw=model(xb); p=final_probability(bb,raw,float(config["train_cap"])); loss=torch.mean((p-yb)**2)
            if not torch.isfinite(loss): raise FloatingPointError(f"NaN/inf loss: {stage} seed={seed} epoch={epoch}")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); total+=float(loss.detach())*len(xb)
        rss=rss_mib(config); gpu_peak=torch.cuda.max_memory_allocated()/1024/1024 if torch.cuda.is_available() else 0
        config["_completed_epoch_units"]=int(config.get("_completed_epoch_units",0))+1
        elapsed=time.monotonic()-float(config.get("_run_started_monotonic",time.monotonic()))
        completed_units=max(int(config["_completed_epoch_units"]),1); total_units=max(int(config.get("_total_epoch_units",completed_units)),completed_units)
        eta=elapsed/completed_units*(total_units-completed_units); projected=(elapsed+eta)/3600
        print(f"{stage} seed={seed} epoch={epoch} brier={total/len(dataset):.9f} seconds={time.perf_counter()-epoch_started:.1f} rss_mib={rss:.1f} gpu_peak_mib={gpu_peak:.1f} eta_hours={eta/3600:.2f} projected_total_hours={projected:.2f} cpu_threads={torch.get_num_threads()}",flush=True)
        if projected>=20: print("WARNING: 현재 속도 기준 전체 예상시간이 20시간 이상입니다.",flush=True)
        raw_pred=None
        if epoch in SNAPSHOTS:
            raw_pred=predict_raw(model,x_predict,device,int(config["batch_size"])); snapshot_predictions.append(raw_pred)
        checkpoint_started=time.perf_counter()
        atomic_torch(completed.parent/f"epoch_{epoch}.pt",{"state_dict":model.state_dict(),"optimizer":optimizer.state_dict(),"generator_state":generator.get_state(),"torch_rng_state":torch.get_rng_state(),"numpy_rng_state":np.random.get_state(),"input_dim":x_train.shape[1],"seed":seed,"epoch":epoch,"prediction":raw_pred,"train_row_id_sha256":train_digest,"predict_row_id_sha256":predict_digest,"training_signature":signature})
        print(f"{stage} seed={seed} epoch={epoch} checkpoint_seconds={time.perf_counter()-checkpoint_started:.2f}",flush=True)
        deadline=config.get("_deadline_monotonic")
        if deadline is not None and time.monotonic()>=float(deadline):
            raise TimeoutError(f"--max-hours 한도 도달: epoch {epoch} 원자 checkpoint 저장 후 안전 종료")
    if len(snapshot_predictions)!=len(SNAPSHOTS): raise RuntimeError(f"snapshot 3/4/5 완성 실패: {stage} seed={seed}")
    raw_mean=np.mean(snapshot_predictions,axis=0)
    temporary=raw_path.with_suffix(".npy.tmp")
    with temporary.open("wb") as stream: np.save(stream,raw_mean)
    os.replace(temporary,raw_path)
    atomic_json(completed,{"status":"complete","rows":len(predict),"train_row_id_sha256":train_digest,"predict_row_id_sha256":predict_digest,"training_signature":signature,"seconds":time.perf_counter()-started})
    return raw_mean


def inner_frames(frame: pd.DataFrame, outer_year: int) -> tuple[pd.DataFrame,pd.DataFrame]:
    if outer_year==2023:
        source=frame.loc[frame.fold.eq(2022)].copy(); months=sorted(source.game_month.dropna().unique())
        if len(months)<2: raise ValueError("2022 내부 temporal split에 필요한 game_month가 부족합니다")
        return source.loc[source.game_month.lt(months[-1])],source.loc[source.game_month.eq(months[-1])]
    return frame.loc[frame.fold.eq(2022)].copy(),frame.loc[frame.fold.eq(2023)].copy()


def choose_cap(validation: pd.DataFrame, raw: np.ndarray) -> tuple[float,list[dict]]:
    records=[]
    for cap in CAPS:
        p,_=numpy_probability(validation.p_915.to_numpy(),raw,cap)
        records.append({"cap":cap,"brier":float(np.mean((p-validation.target.to_numpy())**2))})
    return float(min(records,key=lambda x:(x["brier"],x["cap"]))["cap"]),records


def run_crossfit(frame: pd.DataFrame, seeds: list[int], output_dir: Path, checkpoint_dir: Path,
                 device: torch.device, config: dict) -> dict:
    outputs=[]; selected={}; selection_records=[]; direction_agreement=[]
    for outer_year,train_years in [(2023,[2022]),(2024,[2022,2023])]:
        fold_started=time.perf_counter()
        inner_train,inner_val=inner_frames(frame,outer_year)
        if set(inner_train[ID]) & set(inner_val[ID]): raise ValueError("inner train/validation row overlap")
        inner_raw=[]
        for seed in seeds:
            inner_raw.append(fit_snapshots(inner_train,inner_val,seed,f"outer_{outer_year}_inner",checkpoint_dir,device,config))
        cap,records=choose_cap(inner_val,np.mean(inner_raw,axis=0)); selected[outer_year]=cap
        for rec in records: selection_records.append({"outer_fold":outer_year,"selection_train_rows":len(inner_train),"selection_validation_rows":len(inner_val),**rec})
        del inner_raw
        gc.collect()
        outer_train=frame.loc[frame.fold.isin(train_years)].copy(); outer_eval=frame.loc[frame.fold.eq(outer_year)].copy()
        if set(outer_train[ID]) & set(outer_eval[ID]): raise ValueError("outer train/evaluation row overlap")
        raw=[]
        for seed in seeds:
            raw.append(fit_snapshots(outer_train,outer_eval,seed,f"outer_{outer_year}_refit",checkpoint_dir,device,config))
        if len(raw)>1:
            signs=np.sign(np.stack(raw)); direction_agreement.append(float(np.mean(np.all(signs==signs[0],axis=0))))
        outer_eval["raw_correction"]=np.mean(raw,axis=0)
        outputs.append(outer_eval[[ID,"fold","target","p_915","asof_pitcher_n","game_type","raw_correction"]].copy())
        del inner_train,inner_val,outer_train,outer_eval,raw
        gc.collect()
        print(f"outer_fold={outer_year} total_seconds={time.perf_counter()-fold_started:.1f} rss_mib={rss_mib(config):.1f}",flush=True)
    pd.DataFrame(selection_records).to_csv(output_dir/"cap_selection_history.csv",index=False)
    result=evaluate(pd.concat(outputs,ignore_index=True),selected,output_dir)
    fold_metrics=pd.read_csv(output_dir/"metrics_by_fold.csv"); overall_metrics=pd.read_csv(output_dir/"metrics_overall.csv")
    deploy_folds=fold_metrics.loc[fold_metrics.selection.eq("deployable")]
    result["deployable_fold_deltas"]={str(int(row.fold)):float(row.delta_vs_915) for row in deploy_folds.itertuples()}
    result["deployable_overall_delta"]=float(overall_metrics.loc[overall_metrics.candidate.eq("deployable"),"delta_vs_915"].iloc[0])
    result["seed_direction_agreement"]=direction_agreement
    if len(seeds)>1:
        keep=bool((deploy_folds.delta_vs_915<=0).all() and result["deployable_overall_delta"]<=-float(config["acceptance"]["three_seed_min_overall_improvement"]) and min(direction_agreement,default=1.0)>=0.55)
        result["three_seed_keep_candidate"]=keep; result["decision"]="CONDITIONAL" if keep else "REJECT"
    result["seeds"]=seeds; result["cuda"]={"available":torch.cuda.is_available(),"device":str(device),"name":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"torch":torch.__version__,"peak_allocated_bytes":torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,"peak_reserved_bytes":torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0}
    atomic_json(output_dir/"report.json",result); return result


def load_completed_prediction(checkpoint_dir: Path, stage: str, seed: int,
                              predict: pd.DataFrame, config: dict) -> np.ndarray:
    root=checkpoint_dir/stage/f"seed_{seed}"; marker_path=root/"complete.json"; raw_path=root/"raw_prediction.npy"
    if not marker_path.is_file() or not raw_path.is_file():
        raise FileNotFoundError(f"evaluate에 필요한 완료 checkpoint 누락: {root}")
    marker=json.loads(marker_path.read_text(encoding="utf-8")); raw=np.load(raw_path,allow_pickle=False)
    signature=marker.get("training_signature",{})
    if (marker.get("status")!="complete" or marker.get("predict_row_id_sha256")!=row_digest(predict)
            or signature.get("_config_sha256")!=config.get("_config_sha256")
            or signature.get("_input_asset_sha256")!=config.get("_input_asset_sha256")
            or raw.shape!=(len(predict),) or not np.isfinite(raw).all()):
        raise ValueError(f"evaluate checkpoint 계약 불일치: {root}")
    return raw


def run_evaluate_only(frame: pd.DataFrame, seeds: list[int], output_dir: Path,
                      checkpoint_dir: Path, config: dict) -> dict:
    outputs=[]; selected={}; selection=[]
    for outer_year,train_years in [(2023,[2022]),(2024,[2022,2023])]:
        inner_train,inner_val=inner_frames(frame,outer_year)
        inner_raw=[load_completed_prediction(checkpoint_dir,f"outer_{outer_year}_inner",seed,inner_val,config) for seed in seeds]
        cap,records=choose_cap(inner_val,np.mean(inner_raw,axis=0)); selected[outer_year]=cap
        for record in records: selection.append({"outer_fold":outer_year,"selection_train_rows":len(inner_train),"selection_validation_rows":len(inner_val),**record})
        outer_eval=frame.loc[frame.fold.eq(outer_year)].copy()
        raw=[load_completed_prediction(checkpoint_dir,f"outer_{outer_year}_refit",seed,outer_eval,config) for seed in seeds]
        outer_eval["raw_correction"]=np.mean(raw,axis=0)
        outputs.append(outer_eval[[ID,"fold","target","p_915","asof_pitcher_n","game_type","raw_correction"]].copy())
    pd.DataFrame(selection).to_csv(output_dir/"cap_selection_history.csv",index=False)
    result=evaluate(pd.concat(outputs,ignore_index=True),selected,output_dir)
    result["seeds"]=seeds; result["evaluation_resumed_from_completed_checkpoints"]=True
    atomic_json(output_dir/"report.json",result)
    atomic_json(checkpoint_dir/"evaluation_complete.json",{"status":"complete","config_sha256":config.get("_config_sha256"),"input_asset_sha256":config.get("_input_asset_sha256"),"selected_caps":result["selected_caps"],"report":str((output_dir/'report.json').resolve())})
    return result


def run_final(frame: pd.DataFrame, test: pd.DataFrame, seeds: list[int], cap: float,
              output_dir: Path, checkpoint_dir: Path, device: torch.device, config: dict) -> dict:
    if "target" in test.columns: raise ValueError("final test frame에 target이 존재합니다")
    raw=[]
    for seed in seeds: raw.append(fit_snapshots(frame,test,seed,"final_train",checkpoint_dir,device,config))
    raw_mean=np.mean(raw,axis=0); probability,correction=numpy_probability(test.p_915.to_numpy(),raw_mean,cap)
    if not np.isfinite(probability).all() or not ((probability>=0)&(probability<=1)).all(): raise ValueError("final probability 범위/유한성 실패")
    submission=pd.DataFrame({ID:test[ID].to_numpy(),"control_success":probability})
    if not np.array_equal(submission[ID].to_numpy(),test[ID].to_numpy()): raise ValueError("final row_id 순서 실패")
    submission.to_csv(output_dir/"submission.csv",index=False)
    pd.DataFrame({ID:test[ID],"p_915":test.p_915,"raw_correction":raw_mean,"correction":correction,"control_success":probability}).to_csv(output_dir/"test_residual_predictions.csv.gz",index=False,compression="gzip")
    return {"status":"completed","phase":"final_train","selected_cap":cap,"seeds":seeds,"rows":len(test),"probability_min":float(probability.min()),"probability_max":float(probability.max())}


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--phase",choices=["seed42","three_seed","final_train"],required=True)
    parser.add_argument("--data-root",required=True); parser.add_argument("--asset-root",required=True); parser.add_argument("--mlp-oof-path"); parser.add_argument("--output-dir",required=True); parser.add_argument("--checkpoint-dir",required=True)
    parser.add_argument("--smoke",action="store_true"); parser.add_argument("--local-cpu",action="store_true")
    parser.add_argument("--cpu-workers",type=int,default=1); parser.add_argument("--max-hours",type=float); parser.add_argument("--memory-reserve-gb",type=float,default=4.0); parser.add_argument("--evaluate-only",action="store_true"); args=parser.parse_args()
    if args.cpu_workers<1: raise ValueError("--cpu-workers는 1 이상이어야 합니다")
    if args.max_hours is not None and args.max_hours<=0: raise ValueError("--max-hours는 0보다 커야 합니다")
    output=Path(args.output_dir); checkpoints=Path(args.checkpoint_dir); output.mkdir(parents=True,exist_ok=True); checkpoints.mkdir(parents=True,exist_ok=True)
    config_path=experiment_root()/"configs/experiment.json"; config=load_json(config_path)
    if not args.smoke and not args.local_cpu and not torch.cuda.is_available(): raise RuntimeError("040 GPU full mode는 CUDA GPU가 필수입니다")
    device=torch.device("cpu" if args.smoke or args.local_cpu else "cuda")
    if args.local_cpu:
        torch.set_num_threads(args.cpu_workers)
        try: torch.set_num_interop_threads(1)
        except RuntimeError: pass
    mlp_path=Path(args.mlp_oof_path) if args.mlp_oof_path else None
    asset_report=inspect(Path(args.asset_root),Path(args.data_root),require_final=args.phase=="final_train",mlp_oof_path=mlp_path)
    config["_config_sha256"]=sha256(config_path)
    config["_input_asset_sha256"]=hashlib.sha256(json.dumps({k:v["sha256"] for k,v in asset_report["checked_assets"].items()},sort_keys=True).encode()).hexdigest()
    config["_run_started_monotonic"]=time.monotonic(); config["_deadline_monotonic"]=(time.monotonic()+args.max_hours*3600) if args.max_hours else None
    config["_completed_epoch_units"]=0; config["_peak_rss_mib"]=rss_mib()
    if args.phase=="final_train":
        validation=checkpoints/"three_seed"/"report.json"
        if not validation.is_file(): validation=checkpoints/"seed42"/"report.json"
        if not validation.is_file(): raise PermissionError("final_train 차단: 검증 통과 report가 없습니다")
        prior=json.loads(validation.read_text(encoding="utf-8"))
        if prior.get("decision") not in {"CONDITIONAL","ACCEPT"}: raise PermissionError("final_train 차단: 검증 판정이 통과가 아닙니다")
        frame=build_frame(Path(args.data_root),Path(args.asset_root),mlp_path); test=build_test_frame(Path(args.data_root),Path(args.asset_root)); cap=float(prior["selected_caps"]["2024"])
        seeds=prior.get("seeds",[42]); config["_total_epoch_units"]=len(seeds)*int(config["epochs"]); memory_estimate=estimate_memory(frame,config,args.memory_reserve_gb,enforce=args.local_cpu)
        report=run_final(frame,test,seeds,cap,output,checkpoints/args.phase,device,config); report["memory_estimate"]=memory_estimate; report["runtime"]={"device":str(device),"cpu_threads":torch.get_num_threads(),"peak_rss_mib":config["_peak_rss_mib"]}; atomic_json(output/"report.json",report); atomic_json(checkpoints/args.phase/"report.json",report); return
    prior_seed42=None
    if args.phase=="three_seed":
        prior=checkpoints/"seed42"/"report.json"
        if not prior.is_file() or not json.loads(prior.read_text(encoding="utf-8")).get("seed42_extend_three_seed"):
            raise PermissionError("three_seed 차단: seed42 통과 marker가 없습니다")
        prior_seed42=json.loads(prior.read_text(encoding="utf-8"))
    frame=build_frame(Path(args.data_root),Path(args.asset_root),mlp_path)
    if args.smoke: frame=pd.concat([frame.loc[frame.fold.eq(y)].head(1024) for y in [2022,2023,2024]],ignore_index=True)
    seeds=config["seeds"][args.phase]
    config["_total_epoch_units"]=len(seeds)*4*int(config["epochs"])
    memory_estimate=estimate_memory(frame,config,args.memory_reserve_gb,enforce=args.local_cpu)
    if args.evaluate_only:
        report=run_evaluate_only(frame,seeds,output,checkpoints/args.phase,config); report["memory_estimate"]=memory_estimate; atomic_json(output/"report.json",report); return
    report=run_crossfit(frame,seeds,output,checkpoints/args.phase,device,config)
    if args.phase=="three_seed" and prior_seed42 is not None:
        current=np.asarray(list(report["deployable_fold_deltas"].values())); previous=np.asarray(list(prior_seed42["deployable_fold_deltas"].values()))
        stable=bool(np.std(current)<=np.std(previous)+1e-12); report["more_stable_than_seed42"]=stable
        report["three_seed_keep_candidate"]=bool(report.get("three_seed_keep_candidate") and stable); report["decision"]="CONDITIONAL" if report["three_seed_keep_candidate"] else "REJECT"
    report["runtime"]={"python":platform.python_version(),"platform":platform.platform(),"device":str(device),"cpu_threads":torch.get_num_threads(),"peak_rss_mib":config["_peak_rss_mib"]}
    report["memory_estimate"]=memory_estimate
    atomic_json(output/"report.json",report); atomic_json(checkpoints/args.phase/"report.json",report)
    atomic_json(checkpoints/args.phase/"evaluation_complete.json",{"status":"complete","config_sha256":config.get("_config_sha256"),"input_asset_sha256":config.get("_input_asset_sha256"),"selected_caps":report.get("selected_caps"),"report":str((output/'report.json').resolve())})


if __name__=="__main__":
    try:
        main()
    except (KeyboardInterrupt,TimeoutError) as exc:
        payload={"status":"interrupted","reason":type(exc).__name__,"message":str(exc),"safe_resume":True,"timestamp":time.strftime("%Y-%m-%d %H:%M:%S")}
        for option in ["--output-dir","--checkpoint-dir"]:
            if option in os.sys.argv:
                root=Path(os.sys.argv[os.sys.argv.index(option)+1]); root.mkdir(parents=True,exist_ok=True); atomic_json(root/"INTERRUPTED.json",payload)
        print(f"안전 중단: {type(exc).__name__}: {exc}. 같은 명령으로 resume하십시오.",flush=True)
        raise SystemExit(130)
