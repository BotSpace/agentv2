# Bot Agent

Python + LangGraph asosidagi CLI/API agent. U foydalanuvchi talabi asosida oddiy Telegram bot loyihasini yaratadi.

Generated loyiha odatda quyidagi fayllardan iborat bo'ladi:

- `go.mod`
- `cmd/bot/main.go`
- `internal/app/app.go`
- `internal/config/config.go`
- `internal/handlers/handler.go`
- `internal/state/store.go`
- `pkg/messages/messages.go`
- `.env.example`
- `README.md`

## CLI

```bash
bot-agent chat --project-dir generated_bots/support
```

Debug log bilan:

```bash
BOT_AGENT_DEBUG=true bot-agent chat --project-dir generated_bots/support
```

OpenAI yoki OpenAI-compatible custom API bilan ishlatish:

```bash
OPENAI_API_KEY=sk-... \
OPENAI_BASE_URL=https://your-openai-compatible-api.example/v1 \
BOT_AGENT_MODEL_PROVIDER=openai \
bot-agent chat --model gpt-4.1 --project-dir generated_bots/support
```

Yoki flag orqali:

```bash
bot-agent chat \
  --provider openai \
  --api-key sk-... \
  --api-base https://your-openai-compatible-api.example/v1 \
  --request-timeout 120 \
  --model gpt-4.1 \
  --project-dir generated_bots/support
```

Eslatma: OpenAI-compatible endpoint odatda `/v1` bilan tugaydi. Agar `OPENAI_BASE_URL=http://localhost:4141/` kabi root URL bersangiz, agent uni avtomatik `.../v1` ga to'g'rilaydi.

OpenAI-compatible lokal API sekin javob bersa, `--request-timeout` qiymati server javob vaqtidan katta bo'lishi kerak. Masalan server 25s da javob qaytarsa `--request-timeout 60` yoki `--request-timeout 120` ishlating. Juda kichik timeout client requestni erta yopadi, server logida esa keyinroq `200` ko'rinishi mumkin.

## API

```bash
bot-agent api --project-dir generated_bots/bot --host 127.0.0.1 --port 8000
```

API server uchun ham xuddi shu provider sozlamalari ishlaydi:

```bash
bot-agent api \
  --provider openai \
  --api-key sk-... \
  --api-base https://your-openai-compatible-api.example/v1 \
  --model gpt-4.1
```
