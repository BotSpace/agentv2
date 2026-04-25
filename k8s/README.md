# Kubernetes Deploy

This directory contains Kubernetes manifests for the Botmother Go engine.

## Files

- `base/` contains the reusable deployment, service, ingress, Redis, config, and flow ConfigMap.
- `base/secret.example.yaml` is a template only. Copy it and fill real values locally.
- `overlays/local/` is a local Kustomize overlay with the default image name, local host, and a placeholder flow.

## Build Image

```bash
docker build -t botmother-engine-go:latest .
```

For `kind`:

```bash
kind load docker-image botmother-engine-go:latest
```

For a remote cluster, tag and push the image, then update `agent/k8s/overlays/local/kustomization.yaml`.

## Create Secret

```bash
cp agent/k8s/base/secret.example.yaml /tmp/botmother-engine-secret.yaml
```

Edit `/tmp/botmother-engine-secret.yaml`, then apply it:

```bash
kubectl apply -f agent/k8s/base/namespace.yaml
kubectl apply -f /tmp/botmother-engine-secret.yaml
```

Required values:

- `BOT_TOKEN` or `RELEASE_BOT_TOKEN`
- `PROJECT_ID` if resource monitoring or external webhook URLs are used
- `MONGO_ADDR` and `MONGO_DB_NAME` if collection nodes are used
- `PLUGIN_API_KEY` if the plugin service requires auth

## Deploy

```bash
kubectl apply -k agent/k8s/overlays/local
```

Check status:

```bash
kubectl -n botmother-engine get pods
kubectl -n botmother-engine logs deploy/botmother-engine -f
```

Port-forward locally:

```bash
kubectl -n botmother-engine port-forward svc/botmother-engine 8443:8443 9090:9090
```

Health check:

```bash
curl http://127.0.0.1:9090/health
```

Metrics:

```bash
curl http://127.0.0.1:9090/metrics
```

## Flow Updates

The flow is mounted from the `botmother-flow` ConfigMap at `/app/assets/flow.json`.
The local overlay starts with a placeholder flow so plain `kubectl apply -k` works everywhere.
Replace it with the project flow after deploy:

```bash
kubectl -n botmother-engine create configmap botmother-flow \
  --from-file=flow.json=assets/flow.json \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl -n botmother-engine rollout restart deploy/botmother-engine
```

## Webhooks

The engine exposes:

- `8443` for Telegram webhook mode and `/custom-webhook/...`
- `9090` for `/health` and `/metrics`

Set these env vars when using Telegram webhook mode:

- `WEBHOOK_ENABLED=true`
- `WEBHOOK_URL=https://your-domain.example`
- `WEBHOOK_PORT=8443`

Set `EXTERNAL_WEBHOOK_BASE_URL` when custom code needs generated external webhook URLs.
