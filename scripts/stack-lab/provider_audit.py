#!/usr/bin/env python3
"""PF.10 test-only capability and fallback audit; never changes production output."""
from __future__ import annotations
import importlib.util,json,os,re,shutil,subprocess,sys
from pathlib import Path
ROOT=Path('/Users/dp/Desktop/raw-editor');SOURCE=ROOT/'src-tauri/src/panorama_utils/stack_pipeline/feature_provider.rs';JOINT=ROOT/'output/acceptance/langyuan-v44-joint-field/candidate-repeat/langyuan-10/planes/joint-production-input.json'
def python_module(name):
 try:
  spec=importlib.util.find_spec(name);return {'available':spec is not None,'origin':str(spec.origin) if spec else None}
 except Exception as e:return {'available':False,'error':type(e).__name__}
def command(name):
 p=shutil.which(name);return {'available':p is not None,'path':p}
def main():
 rust=SOURCE.read_text();joint=json.loads(JOINT.read_text())
 cv2=python_module('cv2');ort=python_module('onnxruntime')
 providers={'rust_fast_brief':{'available':True,'kind':'native_default','source':str(SOURCE),'production_default':True},'python_opencv_sift':{'available':cv2['available'],'python_cv2':cv2,'production_default':False},'python_opencv_akaze':{'available':cv2['available'],'python_cv2':cv2,'production_default':False},'python_opencv_ecc':{'available':cv2['available'],'python_cv2':cv2,'production_default':False},'lightglue_disk_aliked':{'available':ort['available'],'onnxruntime':ort,'production_default':False},'roma':{'available':False,'reason':'no packaged provider or weights; explicitly out of this audit'}}
 boundary={'schema':1,'rust_constants':{'schema':int(re.search(r'FEATURE_PROVIDER_SCHEMA: u32 = (\d+)',rust).group(1)),'default':re.search(r'RUST_FAST_BRIEF_PROVIDER: &str = "([^"]+)"',rust).group(1),'sift':re.search(r'PYTHON_OPENCV_SIFT_PROVIDER: &str = "([^"]+)"',rust).group(1)},'fallback_codes':sorted(set(re.findall(r'=> "([a-z_]+)"',rust))),'optional_resolution':'ProviderResolution.resolve(optional) selects rust_fast_brief with python_unavailable when sidecar unavailable','budget_contract':'ProviderThreadBudget validates rayon_threads + opencv_threads <= total_threads and requires each nonzero'}
 result={'schema':1,'task':8,'status':'capability_audit_incomplete_provider_comparison','method':'optional_provider_capability_and_fallback_audit_v1','production_impact':'test_only_no_provider_selected_no_stack_report_or_threshold_change','providers':providers,'boundary':boundary,'langyuan_0_2_reuse':{'path':str(JOINT),'records':[{'left':r['left'],'right':r['right'],'inliers':r['inliers'],'error':r['error'],'hull_support':r.get('diagnostic',{}).get('hull_support')} for r in joint['records']],'note':'saved native Rust-path evidence; no provider rerun'},'tools':{'python':sys.executable,'ffmpeg':command('ffmpeg'),'cargo':command('cargo')},'decision':{'sift_akaze_ecc_measured':False,'learned_provider_measured':False,'reason':'optional dependencies are absent in this environment; no fabricated scores','safe_action':'retain Rust FAST/BRIEF default and record fallback reason; install a pinned provider bundle before PF.10 quality comparison'},'source_sha256':subprocess.check_output(['shasum','-a','256',str(SOURCE)],text=True).split()[0]}
 out=Path(os.environ.get('PROVIDER_AUDIT_OUT','output/acceptance/task-8/provider-capability-audit.json')).resolve();out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n');print(json.dumps({'status':result['status'],'providers':{k:v['available'] for k,v in providers.items()},'output':str(out)},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
