import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SECONDARY_LAUNCHER = Path(__file__).resolve().parents[1] / "tools" / "remote" / "reu_secondary_job.py"
if not SECONDARY_LAUNCHER.is_file():
    raise unittest.SkipTest("Local secondary launcher is unavailable")

from tools.remote import reu_secondary_job as job


class SecondaryJobTests(unittest.TestCase):
    def processes(self):
        return {10:dict(parent=1,args=['python','/release/tools/remote/reu_remote_job.py']),
            11:dict(parent=10,args=['python','main.py','experiment','original-tuning']),
            12:dict(parent=11,args=['python','worker']),20:dict(parent=1,args=['python','secondary'])}

    def test_primary_and_workers_allowed_but_unrelated_python_rejected(self):
        rows=self.processes();meta=dict(pid=11,command=rows[11]['args'])
        job.check_process_scope(meta,rows,20)
        rows[30]=dict(parent=1,args=['python','unrelated'])
        with self.assertRaisesRegex(RuntimeError,'Unexpected'):job.check_process_scope(meta,rows,20)

    def test_stale_pid_and_unprotected_primary_rejected(self):
        rows=self.processes();meta=dict(pid=11,command=['python','changed'])
        with self.assertRaisesRegex(RuntimeError,'command changed'):job.check_process_scope(meta,rows,20)
        rows.pop(10);meta['command']=rows[11]['args']
        with self.assertRaisesRegex(RuntimeError,'launcher is missing'):job.check_process_scope(meta,rows,20)

    def test_only_explicit_secondary_tmux_session_allowed(self):
        with patch.dict(os.environ,{'TMUX':'socket','TMUX_PANE':'%9'}),patch.object(job.subprocess,'check_output',return_value='sugar_run_2\n'):
            self.assertEqual(job.require_secondary_session(),'sugar_run_2')
        with patch.dict(os.environ,{'TMUX':'socket','TMUX_PANE':'%9'}),patch.object(job.subprocess,'check_output',return_value='sugar_run\n'):
            with self.assertRaises(RuntimeError):job.require_secondary_session()

    def test_only_known_live_ic50_record_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            home=Path(tmp).resolve();record=home/job.PRIMARY;record.mkdir(parents=True)
            meta=dict(status='running',gpu='0',tmux_session='sugar_run',command=['python','original-tuning','--family','ic50'])
            (record/'launch.json').write_text(json.dumps(meta));self.assertEqual(job.primary_metadata(record,home),meta)
            meta['status']='succeeded';(record/'launch.json').write_text(json.dumps(meta))
            with self.assertRaises(ValueError):job.primary_metadata(record,home)
            with self.assertRaises(ValueError):job.primary_metadata(home/'other',home)


if __name__=='__main__':unittest.main()
