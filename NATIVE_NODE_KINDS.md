# Native Node Kinds

`agent/` endi runtime’dagi barcha node turlarini native DSL kind sifatida taniydi. Yangi node qo‘shish yoki mavjud kind maydonlarini ko‘rish uchun CLI chat ichida `describe_step_kind(kind)` tool ishlatiladi.

Asosiy guruhlar:

- Triggerlar: `command_trigger`, `message_trigger`, `callback_query_trigger`, `callback_button_trigger`, `reply_button_trigger`, `external_webhook_trigger`, `cron_trigger`
- Xabar/media: `send_text`, `send_photo`, `send_video`, `send_audio`, `send_file`, `send_animation`, `send_voice`, `send_video_note`, `send_location`, `send_contact`, `send_poll`, `send_sticker`, `send_media_group`, `send_venue`, `send_dice`
- Xabar operatsiyalari: `edit_message`, `edit_or_send_text`, `delete_message`, `forward_message`, `copy_message`, `pin_message`, `unpin_message`, `unpin_all_messages`, `chat_action`, `callback_query_answer`, `answer_inline_query`
- Logic va control flow: `check_membership`, `if_condition`, `else_if`, `else`, `delay`, `scheduler`, `random`, `variable`, `for_loop`, `for_loop_continue`, `custom_code`, `subflow`, `subflow_exit`
- Data va integration: `state`, `collection`, `load_collection_item`, `load_collection_list`, `update_collection`, `delete_collection`, `download`, `http_request`, `send_to_admin`, `cron`

Qoida:

- Authoring source har doim DSL.
- Runtime JSON faqat compile natijasi.
- `raw_node` saqlanadi, lekin normal ish jarayonida native kind’lar ishlatiladi.
