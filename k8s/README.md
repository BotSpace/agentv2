# Agentv2 Kubernetes Deploy

This directory deploys the Python Botmother Flow Agent API as a separate `agentv2` workload in the existing `botmother` namespace. It does not replace the existing production `botmother-agent` Deployment.

## CI/CD Flow

On every push to `main`, `.github/workflows/deploy.yml`:

1. Builds the Docker image.
2. Pushes it to `registry.iprogrammer.uz/agentv2/botmother-agent:prod-<short-sha>`.
3. Updates `k8s/chart/agentv2/values-prod.yaml` with the new tag.
4. Commits the tag update back to `main` with `[skip ci]`.
5. Argo CD renders the Helm chart at `k8s/chart/agentv2` with `values-prod.yaml` and syncs it into the `botmother` namespace.

## One-Time Setup

Add these GitHub repository secrets:

- `HARBOR_USERNAME`
- `HARBOR_PASSWORD`

Create the Harbor image pull secret in the cluster:

```bash
kubectl -n botmother create secret docker-registry agentv2-harbor-registry \
  --docker-server=registry.iprogrammer.uz \
  --docker-username=admin \
  --docker-password='<secret>'
```

Create the runtime secret from the example and fill real values:

```bash
cp k8s/chart/agentv2/examples/agentv2-secret.example.yaml /tmp/agentv2-secret.yaml
kubectl -n botmother apply -f /tmp/agentv2-secret.yaml
```

Common runtime values:

- `AGENT_JWT_PUBLIC_KEY` and `AGENT_JWT_ALGORITHMS` for bearer-token validation.
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_REGION` when using Bedrock.
- `OLLAMA_HOST` only if `IS_OLLAMA=true`.

Apply the Argo CD Application after Argo CD is healthy:

```bash
kubectl apply -f k8s/argocd/agentv2-application.yaml
```

Argo CD must also have Git access to `git@github.com:BotSpace/agentv2.git`.

Note: the current cluster inspection showed no Argo Applications and an unavailable `argocd-repo-server`. Fix Argo CD health before relying on automatic sync.

## Local Checks

Render manifests:

```bash
helm template agentv2 k8s/chart/agentv2 -f k8s/chart/agentv2/values-prod.yaml --namespace botmother
```

Build image:

```bash
docker build -t registry.iprogrammer.uz/agentv2/botmother-agent:test .
```

Smoke test:

```bash
docker run --rm -p 18000:8000 registry.iprogrammer.uz/agentv2/botmother-agent:test
curl http://127.0.0.1:18000/health
```

## Post-Deploy Verification

```bash
kubectl -n botmother get deploy,svc,ingress agentv2
kubectl -n botmother rollout status deploy/agentv2
kubectl -n botmother get pods -l app.kubernetes.io/name=agentv2
curl https://api.botmother.uz/api/agentv2/health
kubectl -n argocd get application agentv2
```

## Storage

The API stores SQLite state and the editable flow at `/app/data` on the `agentv2-data` PVC. Redis is deployed as `agentv2-redis` for event fan-out.
