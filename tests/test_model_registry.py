"""Simulaciones: ningún servicio MLflow recibe artefactos ni modelos."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from proyecto.models.persistence import PersistenceControls
from proyecto.tracking import model_registry as r
from tests.test_persistence import artifact


@pytest.fixture
def simulated_approved(artifact, monkeypatch):
    # Solo una simulación del contrato aprobado; nunca guardar como real este modelo sintético.
    X, *_, bundle = artifact
    metadata = deepcopy(bundle.metadata)
    metadata.update(data_scope="real", status="approved_definitive")
    metadata["decision"].update(status="approved_definitive", frozen_for_evaluation=True,
        definitive_approval=dict(approved=True, approved_by="simulated-reviewer", approved_at="2026-11-06T00:00:00Z", validation_report_id="simulation"))
    monkeypatch.setattr(r, "verify_bundle", lambda *a, **kw: {})
    return X.head(), replace(bundle, metadata=metadata)


class Client:
    def __init__(self, fail=False): self.calls, self.fail = [], fail
    def create_run(self, *a, **kw): self.calls.append(("create", kw)); return NS(info=NS(run_id="synthetic-run"))
    def log_artifact(self, *a, **kw):
        self.calls.append(("artifact", a))
        if self.fail: raise RuntimeError("simulated upload failure")
    def log_metric(self, *a, **kw): self.calls.append(("metric", a))
    def set_terminated(self, *a, **kw): self.calls.append(("terminated", kw["status"]))
    def set_model_version_tag(self, name, version, key, value): self.calls.append(("tag", version, key, value))
    def get_model_version(self, name, version): return NS(tags=self.tags)
    def set_registered_model_alias(self, *a): self.calls.append(("alias", a))


def test_tracking_publication_is_distinct_and_failure_not_success(simulated_approved):
    _, bundle = simulated_approved
    for fail in (False, True):
        client = Client(fail)
        if fail:
            with pytest.raises(RuntimeError):
                r.publish_bundle_artifacts(bundle, experiment_id="simulation", controls=PersistenceControls(publish_real_artifacts=True), client=client)
            assert client.calls[-1] == ("terminated", "FAILED")
        else:
            receipt = r.publish_bundle_artifacts(bundle, experiment_id="simulation", controls=PersistenceControls(publish_real_artifacts=True), client=client)
            assert receipt["registered_version"] is None and receipt["scope"] == "tracking_artifacts"
            assert len([c for c in client.calls if c[0] == "artifact"]) == 4
            assert client.calls[-1] == ("terminated", "FINISHED")


def test_registry_records_exact_version_and_never_promotes_automatically(simulated_approved):
    X, bundle = simulated_approved
    client, calls = Client(), []
    @contextmanager
    def start_run(**kw): yield NS(info=NS(run_id="simulated-run"))
    def log_model(**kw):
        calls.append(kw)
        assert kw["pyfunc_predict_fn"] == "predict_proba"
        return NS(model_uri="models:/simulated-artifact")
    ml = NS(active_run=lambda: None, get_registry_uri=lambda: "databricks-uc",
            models=NS(infer_signature=lambda *a: "signature"), start_run=start_run,
            sklearn=NS(log_model=log_model), log_artifact=lambda *a, **kw: None,
            register_model=lambda uri, name: NS(version="17"), MlflowClient=lambda: client)
    controls = PersistenceControls(register_definitive=True, publish_real_artifacts=True, promote_or_approve=True)
    receipt = r.register_definitive_model(bundle, X, registered_name="catalog.schema.riskvision",
        experiment_id="simulated", source_code_dir=Path(__file__).resolve().parents[1] / "src/proyecto",
        controls=controls, mlflow_module=ml)
    assert receipt["version"] == "17" and not any(c[0] == "alias" for c in client.calls)
    client.tags = {"riskvision.status": "approved_definitive", "riskvision.manifest_sha256": bundle.manifest_sha256}
    r.promote_model_version(bundle, receipt, controls=controls, client=client)
    assert client.calls[-1] == ("alias", ("catalog.schema.riskvision", "champion", "17"))
    client.tags = {}
    with pytest.raises(ValueError): r.promote_model_version(bundle, receipt, controls=controls, client=client)
    with pytest.raises(ValueError): r.register_definitive_model(bundle, X, registered_name="unqualified",
        experiment_id="s", source_code_dir=Path(__file__).resolve().parents[1] / "src/proyecto", controls=controls, mlflow_module=ml)
