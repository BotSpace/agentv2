# Agentv2 Kubernetes Manifests

This directory keeps the Helm chart used to render the Python Botmother Flow Agent API manifests. The live cluster is managed by Flux CD from the separate `BotSpace/k8s` repository.

## CI/CD Flow

On every push to `main`, `.github/workflows/deploy.yml`:

1. Builds the Docker image.
2. Pushes it to `registry.iprogrammer.uz/agentv2/botmother-agent:prod-<short-sha>`.
3. Updates `k8s/chart/agentv2/values-prod.yaml` with the new tag.
4. Commits the tag update back to `main` with `[skip ci]`.

Flux CD does not read this repository directly. Rendered Kubernetes YAML for Flux lives in the `BotSpace/k8s` repo.

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

To deploy through Flux, update the rendered manifest in the `BotSpace/k8s` repository and enable the resource from that repo's Kustomization.

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
kubectl -n flux-system get kustomization
```

## Storage

The API stores SQLite state and the editable flow at `/app/data` on the `agentv2-data` PVC. Redis is deployed as `agentv2-redis` for event fan-out.
