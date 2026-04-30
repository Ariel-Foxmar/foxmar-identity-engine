from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import os
import signal
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import requests


LOG_FORMAT = '%(asctime)s %(levelname)s %(message)s'
logging.basicConfig(level=os.environ.get('LOG_LEVEL', 'INFO').upper(), format=LOG_FORMAT)
log = logging.getLogger('foxmar.identity.worker')

_stop_requested = False


def _handle_stop(signum, frame):  # noqa: ARG001
    global _stop_requested
    _stop_requested = True
    log.info('Stop requested; worker will exit after current cycle.')


signal.signal(signal.SIGTERM, _handle_stop)
signal.signal(signal.SIGINT, _handle_stop)


def _env(name: str, default: str = '') -> str:
    return (os.environ.get(name, default) or '').strip()


def _env_int(name: str, default: int, minimum: Optional[int] = None) -> int:
    try:
        value = int(_env(name, str(default)))
    except ValueError:
        value = int(default)
    if minimum is not None:
        return max(minimum, value)
    return value


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _env(name)
    if not raw:
        return bool(default)
    return raw.lower() in {'1', 'true', 'yes', 'y', 'on'}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _split_csv(value: str) -> List[str]:
    return [part.strip() for part in value.split(',') if part.strip()]


@dataclass(frozen=True)
class WorkerConfig:
    portal_base_url: str
    worker_token: str
    worker_id: str
    engine_mode: str
    poll_interval_seconds: int
    request_timeout_seconds: int
    heartbeat_interval_seconds: int
    job_types: List[str]
    stub_job_result: str
    run_once: bool

    @classmethod
    def from_env(cls) -> 'WorkerConfig':
        return cls(
            portal_base_url=_env('PORTAL_BASE_URL').rstrip('/'),
            worker_token=_env('IDENTITY_WORKER_TOKEN'),
            worker_id=_env('IDENTITY_WORKER_ID', 'unraid-identity-worker'),
            engine_mode=_env('ENGINE_MODE', 'stub').lower(),
            poll_interval_seconds=_env_int('POLL_INTERVAL_SECONDS', 5, minimum=1),
            request_timeout_seconds=_env_int('REQUEST_TIMEOUT_SECONDS', 30, minimum=3),
            heartbeat_interval_seconds=_env_int('HEARTBEAT_INTERVAL_SECONDS', 20, minimum=5),
            job_types=_split_csv(_env('JOB_TYPES', 'enroll_student_photo,reindex_student_photo,verify_face,identify_faces')),
            stub_job_result=_env('STUB_JOB_RESULT', 'fail').lower(),
            run_once=_env_bool('RUN_ONCE', False),
        )

    def validate(self) -> None:
        missing = []
        if not self.portal_base_url:
            missing.append('PORTAL_BASE_URL')
        if not self.worker_token:
            missing.append('IDENTITY_WORKER_TOKEN')
        if missing:
            raise RuntimeError(f'Missing required environment variables: {", ".join(missing)}')


class PortalClient:
    def __init__(self, config: WorkerConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {config.worker_token}',
            'Content-Type': 'application/json',
            'User-Agent': f'foxmar-identity-engine/{config.worker_id}',
            'X-Identity-Worker-ID': config.worker_id,
        })

    def _url(self, path: str) -> str:
        return f'{self.config.portal_base_url}{path}'

    def post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = self.session.post(
            self._url(path),
            data=json.dumps(payload),
            timeout=self.config.request_timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def get(self, path: str) -> Dict[str, Any]:
        response = self.session.get(
            self._url(path),
            timeout=self.config.request_timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def health(self) -> Dict[str, Any]:
        return self.get('/api/v1/identity/health')

    def next_job(self) -> Optional[Dict[str, Any]]:
        payload = {
            'worker_id': self.config.worker_id,
            'job_types': self.config.job_types,
        }
        data = self.post('/api/v1/identity/jobs/next', payload)
        return data.get('job')

    def heartbeat(self, job_uuid: str) -> Dict[str, Any]:
        return self.post(f'/api/v1/identity/jobs/{job_uuid}/heartbeat', {
            'worker_id': self.config.worker_id,
            'heartbeat_utc': _utc_now(),
        })

    def complete(self, job_uuid: str, *, result_payload: Dict[str, Any], templates: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        return self.post(f'/api/v1/identity/jobs/{job_uuid}/complete', {
            'worker_id': self.config.worker_id,
            'result_payload': result_payload,
            'templates': templates or [],
        })

    def fail(self, job_uuid: str, *, error_code: str, error_message: str, result_payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.post(f'/api/v1/identity/jobs/{job_uuid}/fail', {
            'worker_id': self.config.worker_id,
            'error_code': error_code,
            'error_message': error_message,
            'result_payload': result_payload or {},
        })


class IdentityWorker:
    def __init__(self, config: WorkerConfig):
        self.config = config
        self.client = PortalClient(config)

    def log_gpu_status(self) -> None:
        try:
            output = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
                stderr=subprocess.STDOUT,
                text=True,
                timeout=10,
            ).strip()
            log.info('NVIDIA GPU detected: %s', output or 'unknown')
        except Exception as exc:  # pragma: no cover - depends on host GPU runtime
            log.warning('nvidia-smi unavailable inside container: %s', exc)

    def run(self) -> None:
        self.config.validate()
        log.info('Starting Fox-Mar Identity Engine worker id=%s mode=%s portal=%s',
                 self.config.worker_id, self.config.engine_mode, self.config.portal_base_url)
        self.log_gpu_status()

        try:
            health = self.client.health()
            log.info('Portal identity health: %s', json.dumps(health, sort_keys=True))
        except Exception as exc:
            log.warning('Portal health check failed: %s', exc)

        while not _stop_requested:
            try:
                job = self.client.next_job()
                if not job:
                    log.debug('No queued Identity Engine job found.')
                    if self.config.run_once:
                        return
                    time.sleep(self.config.poll_interval_seconds)
                    continue

                self.process_job(job)
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else 'unknown'
                body = exc.response.text[:500] if exc.response is not None else ''
                log.error('Portal HTTP error status=%s body=%s', status, body)
                if self.config.run_once:
                    raise
                time.sleep(self.config.poll_interval_seconds)
            except Exception as exc:
                log.exception('Worker loop error: %s', exc)
                if self.config.run_once:
                    raise
                time.sleep(self.config.poll_interval_seconds)

    def process_job(self, job: Dict[str, Any]) -> None:
        job_uuid = job['job_uuid']
        job_type = job.get('job_type')
        log.info('Claimed job uuid=%s type=%s school=%s student=%s attempt=%s',
                 job_uuid, job_type, job.get('school_id'), job.get('id_student_id'), job.get('attempt_count'))

        try:
            self.client.heartbeat(job_uuid)
            if self.config.engine_mode == 'stub':
                self.process_stub(job)
                return

            self.client.fail(
                job_uuid,
                error_code='engine_mode_not_implemented',
                error_message=f'ENGINE_MODE={self.config.engine_mode} is not implemented in this worker build yet.',
                result_payload={'engine_mode': self.config.engine_mode},
            )
        except Exception as exc:
            log.exception('Job failed unexpectedly uuid=%s: %s', job_uuid, exc)
            try:
                self.client.fail(
                    job_uuid,
                    error_code='worker_exception',
                    error_message=str(exc),
                    result_payload={'engine_mode': self.config.engine_mode},
                )
            except Exception:
                log.exception('Could not report failure for job uuid=%s', job_uuid)

    def process_stub(self, job: Dict[str, Any]) -> None:
        job_uuid = job['job_uuid']
        payload = {
            'engine_mode': 'stub',
            'message': 'Worker handshake succeeded. Real GPU inference is not enabled yet.',
            'worker_id': self.config.worker_id,
        }
        if self.config.stub_job_result == 'complete':
            self.client.complete(job_uuid, result_payload=payload, templates=[])
            log.info('Stub completed job uuid=%s without templates.', job_uuid)
            return

        self.client.fail(
            job_uuid,
            error_code='stub_not_implemented',
            error_message='Worker handshake succeeded. Real GPU inference is not enabled yet.',
            result_payload=payload,
        )
        log.info('Stub failed job uuid=%s intentionally after handshake.', job_uuid)


def main() -> int:
    try:
        config = WorkerConfig.from_env()
        IdentityWorker(config).run()
        return 0
    except Exception as exc:
        log.exception('Fatal worker error: %s', exc)
        return 1


if __name__ == '__main__':
    sys.exit(main())
