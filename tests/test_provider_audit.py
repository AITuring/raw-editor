import json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SCRIPT=ROOT/'scripts/stack-lab/provider_audit.py'
class ProviderAuditTest(unittest.TestCase):
 def test_audit_fails_closed_without_optional_modules(self):
  with tempfile.TemporaryDirectory() as td:
   out=Path(td)/'audit.json';subprocess.run([sys.executable,str(SCRIPT)],cwd=ROOT,env={**os.environ,'PROVIDER_AUDIT_OUT':str(out)},check=True,capture_output=True,text=True);d=json.loads(out.read_text());self.assertEqual(d['status'],'capability_audit_incomplete_provider_comparison');self.assertTrue(d['providers']['rust_fast_brief']['available']);self.assertFalse(d['decision']['sift_akaze_ecc_measured']);self.assertFalse(d['decision']['learned_provider_measured']);self.assertIn('python_unavailable',d['boundary']['fallback_codes']);self.assertTrue(d['providers']['rust_fast_brief']['production_default'])
 def test_report_never_claims_a_provider_score(self):
  text=SCRIPT.read_text();self.assertIn('no fabricated scores',text);self.assertIn('no_provider_selected',text)
if __name__=='__main__':unittest.main()
