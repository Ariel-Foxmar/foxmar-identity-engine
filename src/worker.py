from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import logging
import os
import signal
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import requests


LOG_FORMAT = '%(asctime)s %(levelname)s %(message)s'
logging.basicConfig(level=os.environ.get('LOG_LEVEL', 'INFO').upper(), format=LOG_FORMAT)
log = logging.getLogger('foxmar.identity.worker')

_stop_requested = False


class IdentityEngineError(RuntimeError):
    error_code = 'identity_engine_error'


class NoFaceDetected(IdentityEngineError):
    error_code = 'no_face_detected'


class EmbeddingBackendUnavailable(IdentityEngineError):
    error_code = 'embedding_backend_unavailable'


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
    batch_size: int
    request_timeout_seconds: int
    heartbeat_interval_seconds: int
    job_types: List[str]
    stub_job_result: str
    run_once: bool
    model_dir: str
    insightface_model_name: str
    insightface_det_size: int

    @classmethod
    def from_env(cls) -> 'WorkerConfig':
        return cls(
            portal_base_url=_env('PORTAL_BASE_URL').rstrip('/'),
            worker_token=_env('IDENTITY_WORKER_TOKEN'),
            worker_id=_env('IDENTITY_WORKER_ID', 'unraid-identity-worker'),
            engine_mode=_env('ENGINE_MODE', 'stub').lower(),
            poll_interval_seconds=_env_int('POLL_INTERVAL_SECONDS', 5, minimum=1),
            batch_size=min(_env_int('IDENTITY_WORKER_BATCH_SIZE', 15, minimum=1), 50),
            request_timeout_seconds=_env_int('REQUEST_TIMEOUT_SECONDS', 30, minimum=3),
            heartbeat_interval_seconds=_env_int('HEARTBEAT_INTERVAL_SECONDS', 20, minimum=5),
            job_types=_split_csv(_env('JOB_TYPES', 'enroll_student_photo,reindex_student_photo,verify_face,identify_faces')),
            stub_job_result=_env('STUB_JOB_RESULT', 'fail').lower(),
            run_once=_env_bool('RUN_ONCE', False),
            model_dir=_env('MODEL_DIR', '/models'),
            insightface_model_name=_env('INSIGHTFACE_MODEL_NAME', 'buffalo_l'),
            insightface_det_size=_env_int('INSIGHTFACE_DET_SIZE', 640, minimum=160),
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
            'User-Agent': f'foxmar-identity-engine/{config.worker_id}',
            'X-Identity-Worker-ID': config.worker_id,
        })

    def _url(self, path: str) -> str:
        return f'{self.config.portal_base_url}{path}'

    def post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = self.session.post(
            self._url(path),
            json=payload,
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

    def download_student_photo(self, job_uuid: str) -> Tuple[bytes, Dict[str, str]]:
        response = self.session.get(
            self._url(f'/api/v1/identity/jobs/{job_uuid}/student-photo'),
            timeout=self.config.request_timeout_seconds,
        )
        response.raise_for_status()
        return response.content, dict(response.headers)

    def download_source_image(self, job: Dict[str, Any]) -> Tuple[bytes, Dict[str, str]]:
        job_uuid = job['job_uuid']
        source_path = job.get('source_image_url') or f'/api/v1/identity/jobs/{job_uuid}/source-image'
        url = source_path if source_path.startswith(('http://', 'https://')) else self._url(source_path)
        response = self.session.get(url, timeout=self.config.request_timeout_seconds)
        response.raise_for_status()
        return response.content, dict(response.headers)

    def health(self) -> Dict[str, Any]:
        return self.get('/api/v1/identity/health')

    def next_jobs(self) -> List[Dict[str, Any]]:
        payload = {
            'worker_id': self.config.worker_id,
            'job_types': self.config.job_types,
            'batch_size': self.config.batch_size,
        }
        data = self.post('/api/v1/identity/jobs/next', payload)
        jobs = data.get('jobs')
        if isinstance(jobs, list):
            return [job for job in jobs if job]
        job = data.get('job')
        return [job] if job else []

    def next_job(self) -> Optional[Dict[str, Any]]:
        jobs = self.next_jobs()
        return jobs[0] if jobs else None

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


class InsightFaceBackend:
    def __init__(self, config: WorkerConfig):
        self.config = config
        try:
            import onnxruntime as ort
            from insightface.app import FaceAnalysis
        except Exception as exc:  # pragma: no cover - depends on optional backend install
            raise EmbeddingBackendUnavailable(
                'InsightFace backend is not installed. Rebuild with INSTALL_INSIGHTFACE=true.'
            ) from exc

        available = ort.get_available_providers()
        requested = _split_csv(_env('ONNXRUNTIME_PROVIDERS', 'CUDAExecutionProvider,CPUExecutionProvider'))
        providers = [provider for provider in requested if provider in available]
        if not providers:
            providers = ['CPUExecutionProvider']
        ctx_id = 0 if 'CUDAExecutionProvider' in providers else -1

        log.info('Loading InsightFace model=%s providers=%s available_providers=%s',
                 config.insightface_model_name, providers, available)
        self.app = FaceAnalysis(
            name=config.insightface_model_name,
            root=config.model_dir,
            providers=providers,
        )
        det_size = (config.insightface_det_size, config.insightface_det_size)
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)
        self.embedding_model = f'insightface/{config.insightface_model_name}'
        self.embedding_model_version = _env('INSIGHTFACE_MODEL_VERSION', '0.7.3')
        self.detector_model = f'insightface/{config.insightface_model_name}/detector'
        self.alignment_model = f'insightface/{config.insightface_model_name}/alignment'
        self.providers = providers

    @staticmethod
    def _face_area(face: Any) -> float:
        bbox = getattr(face, 'bbox', None)
        if bbox is None or len(bbox) < 4:
            return 0.0
        return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))

    def _best_face(self, faces: List[Any]) -> Tuple[int, Any]:
        ranked = sorted(
            enumerate(faces),
            key=lambda item: (float(getattr(item[1], 'det_score', 0.0) or 0.0), self._face_area(item[1])),
            reverse=True,
        )
        return ranked[0]

    @staticmethod
    def _embedding_for(face: Any):
        import numpy as np

        embedding = getattr(face, 'normed_embedding', None)
        if embedding is None:
            embedding = getattr(face, 'embedding', None)
        if embedding is None:
            raise IdentityEngineError('InsightFace did not return an embedding for the selected face.')

        vector = np.asarray(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector = vector / norm
        return vector

    @staticmethod
    def _face_box(face: Any) -> Optional[Dict[str, float]]:
        bbox = getattr(face, 'bbox', None)
        if bbox is None or len(bbox) < 4:
            return None
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        return {
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'width': max(0.0, x2 - x1),
            'height': max(0.0, y2 - y1),
        }

    @staticmethod
    def _landmarks(face: Any) -> Optional[List[Dict[str, float]]]:
        kps = getattr(face, 'kps', None)
        if kps is None:
            return None
        return [{'x': float(point[0]), 'y': float(point[1])} for point in kps]

    @staticmethod
    def _pose(face: Any) -> Optional[Dict[str, float]]:
        pose = getattr(face, 'pose', None)
        if pose is None:
            return None
        values = [float(v) for v in pose]
        keys = ['pitch', 'yaw', 'roll']
        return {keys[index] if index < len(keys) else f'value_{index}': value for index, value in enumerate(values)}

    @staticmethod
    def _decode_image(image_bytes: bytes):
        import cv2
        import numpy as np

        if not image_bytes:
            raise IdentityEngineError('Source image download was empty.')
        raw = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if image is None:
            raise IdentityEngineError('Could not decode source image bytes.')
        return image

    def _face_location(self, face: Any) -> Optional[List[int]]:
        box = self._face_box(face)
        if not box:
            return None
        return [
            int(round(box['y1'])),
            int(round(box['x2'])),
            int(round(box['y2'])),
            int(round(box['x1'])),
        ]

    def _face_sort_key(self, face: Any) -> Tuple[float, float]:
        box = self._face_box(face) or {}
        return (float(box.get('y1', 0.0)), float(box.get('x1', 0.0)))

    def enroll_student_photo(self, job: Dict[str, Any], image_bytes: bytes, headers: Dict[str, str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        import cv2
        import numpy as np
        from PIL import Image

        if not image_bytes:
            raise NoFaceDetected('Student photo download was empty.')

        image_hash = hashlib.sha256(image_bytes).hexdigest()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        rgb = np.asarray(image)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        faces = self.app.get(bgr)
        if not faces:
            raise NoFaceDetected('No face was detected in the selected student photo.')

        selected_index, face = self._best_face(faces)
        embedding = self._embedding_for(face)
        confidence = float(getattr(face, 'det_score', 0.0) or 0.0)
        request_payload = job.get('request_payload') or {}
        source_photo_path = headers.get('X-Identity-Source-Photo-Path') or request_payload.get('source_photo_path')

        template = {
            'id_student_id': job.get('id_student_id'),
            'template_type': request_payload.get('template_type') or 'studio_reference',
            'status': 'active',
            'source': 'identity_worker',
            'embedding': [float(value) for value in embedding.tolist()],
            'embedding_model': self.embedding_model,
            'embedding_model_version': self.embedding_model_version,
            'detector_model': self.detector_model,
            'alignment_model': self.alignment_model,
            'vector_metric': 'cosine',
            'quality_score': confidence,
            'source_confidence': confidence,
            'detection_confidence': confidence,
            'face_box': self._face_box(face),
            'landmarks': self._landmarks(face),
            'pose': self._pose(face),
            'image_sha256': image_hash,
            'source_photo_path': source_photo_path,
            'metadata': {
                'engine': 'Identity Engine',
                'backend': 'insightface',
                'providers': self.providers,
                'faces_detected': len(faces),
                'selected_face_index': selected_index,
                'image_width': int(image.width),
                'image_height': int(image.height),
            },
        }
        result = {
            'engine_mode': self.config.engine_mode,
            'backend': 'insightface',
            'worker_id': self.config.worker_id,
            'embedding_model': self.embedding_model,
            'embedding_dim': int(len(embedding)),
            'faces_detected': int(len(faces)),
            'selected_face_index': int(selected_index),
            'detection_confidence': confidence,
            'image_sha256': image_hash,
        }
        return template, result

    def identify_faces(self, job: Dict[str, Any], image_bytes: bytes, headers: Dict[str, str]) -> Dict[str, Any]:
        image_hash = headers.get('X-Identity-Source-Image-SHA256') or hashlib.sha256(image_bytes).hexdigest()
        image = self._decode_image(image_bytes)
        image_height, image_width = image.shape[:2]
        faces = self.app.get(image) or []
        ranked_faces = sorted(enumerate(faces), key=lambda item: self._face_sort_key(item[1]))

        face_payloads: List[Dict[str, Any]] = []
        for output_index, (source_index, face) in enumerate(ranked_faces):
            embedding = self._embedding_for(face)
            confidence = float(getattr(face, 'det_score', 0.0) or 0.0)
            face_payloads.append({
                'face_index': int(output_index),
                'source_face_index': int(source_index),
                'embedding': [float(value) for value in embedding.tolist()],
                'embedding_dim': int(len(embedding)),
                'embedding_model': self.embedding_model,
                'embedding_model_version': self.embedding_model_version,
                'detector_model': self.detector_model,
                'alignment_model': self.alignment_model,
                'vector_metric': 'cosine',
                'detection_confidence': confidence,
                'face_box': self._face_box(face),
                'location': self._face_location(face),
                'landmarks': self._landmarks(face),
                'pose': self._pose(face),
            })

        return {
            'engine_mode': self.config.engine_mode,
            'backend': 'insightface',
            'worker_id': self.config.worker_id,
            'embedding_model': self.embedding_model,
            'embedding_model_version': self.embedding_model_version,
            'embedding_dim': int(len(face_payloads[0]['embedding'])) if face_payloads else 0,
            'detector_model': self.detector_model,
            'alignment_model': self.alignment_model,
            'vector_metric': 'cosine',
            'providers': self.providers,
            'image_sha256': image_hash,
            'image_width': int(image_width),
            'image_height': int(image_height),
            'faces_detected': int(len(faces)),
            'faces_processed': int(len(face_payloads)),
            'faces': face_payloads,
        }

    def warmup(self) -> None:
        """Run one harmless detection pass so provider/model setup happens at startup."""
        import numpy as np

        size = max(160, int(self.config.insightface_det_size or 640))
        image = np.zeros((size, size, 3), dtype=np.uint8)
        self.app.get(image)


class IdentityWorker:
    def __init__(self, config: WorkerConfig):
        self.config = config
        self.client = PortalClient(config)
        self._backend: Optional[InsightFaceBackend] = None

    def backend(self) -> InsightFaceBackend:
        if self._backend is None:
            self._backend = InsightFaceBackend(self.config)
        return self._backend

    def preload_backend_if_needed(self) -> None:
        if self.config.engine_mode not in {'insightface', 'arcface'}:
            return

        started = time.perf_counter()
        log.info('Preloading Identity Engine backend before claiming jobs...')
        backend = self.backend()
        try:
            backend.warmup()
        except Exception as exc:  # pragma: no cover - depends on host inference runtime
            log.warning('Identity Engine backend warmup failed; worker will continue: %s', exc)
        elapsed = time.perf_counter() - started
        log.info('Identity Engine backend ready model=%s providers=%s elapsed=%.2fs',
                 backend.embedding_model, backend.providers, elapsed)

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
        log.info('Starting Identity Engine worker id=%s mode=%s batch_size=%s portal=%s',
                 self.config.worker_id, self.config.engine_mode, self.config.batch_size, self.config.portal_base_url)
        self.log_gpu_status()

        try:
            health = self.client.health()
            log.info('Portal identity health: %s', json.dumps(health, sort_keys=True))
        except Exception as exc:
            log.warning('Portal health check failed: %s', exc)

        self.preload_backend_if_needed()

        while not _stop_requested:
            try:
                jobs = self.client.next_jobs()
                if not jobs:
                    log.debug('No queued Identity Engine job found.')
                    if self.config.run_once:
                        return
                    time.sleep(self.config.poll_interval_seconds)
                    continue

                if len(jobs) > 1:
                    log.info('Claimed Identity Engine batch size=%s', len(jobs))
                for job in jobs:
                    if _stop_requested:
                        break
                    self.process_job(job)
                if self.config.run_once:
                    return
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

            if self.config.engine_mode in {'insightface', 'arcface'}:
                if job_type in {'enroll_student_photo', 'reindex_student_photo'}:
                    self.process_student_photo_enrollment(job)
                    return
                if job_type == 'identify_faces':
                    self.process_identify_faces(job)
                    return
                self.client.fail(
                    job_uuid,
                    error_code='job_type_not_implemented',
                    error_message=f'Job type {job_type} is not implemented by the InsightFace worker yet.',
                    result_payload={'engine_mode': self.config.engine_mode, 'job_type': job_type},
                )
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
                    error_code=getattr(exc, 'error_code', 'worker_exception'),
                    error_message=str(exc),
                    result_payload={'engine_mode': self.config.engine_mode},
                )
            except Exception:
                log.exception('Could not report failure for job uuid=%s', job_uuid)

    def process_student_photo_enrollment(self, job: Dict[str, Any]) -> None:
        job_uuid = job['job_uuid']
        if not job.get('id_student_id'):
            self.client.fail(
                job_uuid,
                error_code='missing_student',
                error_message='Enrollment jobs require id_student_id.',
                result_payload={'engine_mode': self.config.engine_mode},
            )
            return

        try:
            image_bytes, headers = self.client.download_student_photo(job_uuid)
        except requests.HTTPError as exc:
            error_code = 'student_photo_download_failed'
            error_message = str(exc)
            if exc.response is not None:
                try:
                    error_data = exc.response.json()
                    error_code = error_data.get('error') or error_code
                    error_message = error_data.get('message') or error_code
                except ValueError:
                    error_message = exc.response.text[:300] or error_message
            self.client.fail(
                job_uuid,
                error_code=error_code,
                error_message=error_message,
                result_payload={'engine_mode': self.config.engine_mode},
            )
            return

        template, result = self.backend().enroll_student_photo(job, image_bytes, headers)
        self.client.complete(job_uuid, result_payload=result, templates=[template])
        log.info('Completed enrollment job uuid=%s student=%s embedding_dim=%s confidence=%.4f',
                 job_uuid, job.get('id_student_id'), result.get('embedding_dim'), result.get('detection_confidence') or 0.0)

    def process_identify_faces(self, job: Dict[str, Any]) -> None:
        job_uuid = job['job_uuid']
        try:
            image_bytes, headers = self.client.download_source_image(job)
        except requests.HTTPError as exc:
            error_code = 'source_image_download_failed'
            error_message = str(exc)
            if exc.response is not None:
                try:
                    error_data = exc.response.json()
                    error_code = error_data.get('error') or error_code
                    error_message = error_data.get('message') or error_code
                except ValueError:
                    error_message = exc.response.text[:300] or error_message
            self.client.fail(
                job_uuid,
                error_code=error_code,
                error_message=error_message,
                result_payload={'engine_mode': self.config.engine_mode, 'job_type': job.get('job_type')},
            )
            return

        result = self.backend().identify_faces(job, image_bytes, headers)
        self.client.complete(job_uuid, result_payload=result, templates=[])
        log.info('Completed identify_faces job uuid=%s faces=%s embedding_model=%s',
                 job_uuid, result.get('faces_processed'), result.get('embedding_model'))

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
