from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from bm_flow_agent.dsl.catalog import native_kinds


class DSLBaseModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class FlowMeta(DSLBaseModel):
    name: str = "untitled"
    description: str | None = None
    version: int = 1
    tags: list[str] = Field(default_factory=list)


class RouteSpec(DSLBaseModel):
    source: str = Field(alias="from")
    target: str = Field(alias="to")
    on: str | None = None
    source_handle: str | None = None
    target_handle: str | None = None

    @property
    def effective_source_handle(self) -> str | None:
        return self.source_handle or self.on


class ButtonSpec(DSLBaseModel):
    text: str
    type: str = "callback"
    value: str | None = None
    next: str | None = None
    collection_data: dict[str, Any] | None = Field(default=None, alias="collectionData")


class KeyboardSpec(DSLBaseModel):
    active: str = "none"
    inline: list[list[ButtonSpec]] = Field(default_factory=list)
    reply: list[list[ButtonSpec]] = Field(default_factory=list)
    reply_options: dict[str, Any] | None = Field(default=None, alias="replyOptions")


class StepSpec(DSLBaseModel):
    id: str
    kind: str = Field(description=f"Native DSL kind. Supported values include: {', '.join(native_kinds())}, raw_node.")
    title: str | None = None

    command: str | None = None
    global_flag: bool | None = Field(default=None, alias="global")
    with_args: bool | None = None

    message_type: str | None = None
    filter: str | None = None
    value: Any = None
    state: dict[str, Any] | None = None
    buttons: list[dict[str, Any]] = Field(default_factory=list)

    text: Any = None
    parse_mode: str | None = None
    keyboard: KeyboardSpec | dict[str, Any] | None = None
    recipients: dict[str, Any] | None = None
    disable_web_page_preview: bool | None = None
    disable_notification: bool | None = None
    dynamic_messages: list[dict[str, Any]] = Field(default_factory=list)
    target_message_step: str | None = None
    target_node_id: str | None = None
    message_id_source: str | None = None
    message_id: int | None = None
    delete_from_context: bool | None = None
    from_chat_id: int | None = None
    unpin_all: bool | None = None

    action: str | None = None

    state_key: str | None = None
    state_type: str | None = None
    custom_type: str | None = None
    collection_key: str | None = None
    field_name: str | None = None

    collection_name: str | None = None
    context_key: str | None = None
    filters: list[dict[str, Any]] = Field(default_factory=list)
    limit: int | None = None
    skip: int | None = None
    field_mappings: dict[str, Any] | None = None

    operation: str | None = None
    variable_name: str | None = None
    increment: float | None = None
    decrement: float | None = None

    conditions: list[dict[str, Any]] = Field(default_factory=list)
    operator: str | None = None
    branches: list[dict[str, Any]] = Field(default_factory=list)

    code: str | None = None
    lua_code: str | None = None
    timeout: int | float | None = None

    selected_callbacks: list[str] = Field(default_factory=list)
    selected_buttons: list[dict[str, Any]] = Field(default_factory=list)

    provider: str | None = None
    schedule: str | None = None
    enabled: bool | None = None
    target_chat_ids: list[int] = Field(default_factory=list)

    photo_source_type: str | None = None
    photo_file_id: str | None = None
    photo_url: str | None = None
    video_source_type: str | None = None
    video_file_id: str | None = None
    video_url: str | None = None
    audio_source_type: str | None = None
    audio_file_id: str | None = None
    audio_url: str | None = None
    file_source_type: str | None = None
    file_id: str | None = None
    file_url: str | None = None
    animation_source_type: str | None = None
    animation_file_id: str | None = None
    animation_url: str | None = None
    voice_source_type: str | None = None
    voice_file_id: str | None = None
    voice_url: str | None = None
    video_note_source_type: str | None = None
    video_note_file_id: str | None = None
    video_note_url: str | None = None
    sticker_source_type: str | None = None
    sticker_file_id: str | None = None
    sticker_url: str | None = None
    uploaded_file_uuid: str | None = None
    caption: str | None = None
    duration: int | float | None = None
    length: int | None = None

    latitude: float | None = None
    longitude: float | None = None
    live_period: int | None = None
    heading: int | None = None
    horizontal_accuracy: float | None = None
    proximity_alert_radius: int | None = None
    phone_number: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    v_card: str | None = Field(default=None, alias="vCard")
    question: str | None = None
    options: list[Any] = Field(default_factory=list)
    is_anonymous: bool | None = None
    allows_multiple_answers: bool | None = None
    correct_option_id: int | None = None
    explanation: str | None = None
    explanation_parse_mode: str | None = None
    open_period: int | None = None
    is_closed: bool | None = None
    media_items: list[dict[str, Any]] = Field(default_factory=list)
    address: str | None = None
    foursquare_id: str | None = None
    foursquare_type: str | None = None
    emoji: str | None = None

    callback_query_id: str | None = None
    show_alert: bool | None = None
    url: str | None = None
    cache_time: int | None = None
    results: list[dict[str, Any]] = Field(default_factory=list)
    is_personal: bool | None = None
    channels: list[dict[str, Any]] = Field(default_factory=list)
    channel_id: str | int | None = None
    channel_username: str | None = None

    delay: int | float | None = None
    delay_seconds: int | float | None = None
    scheduled_date_time: str | None = None
    date: str | None = None
    time: str | None = None
    timestamp: int | float | None = None
    year: int | None = None
    month: int | None = None
    day: int | None = None
    hour: int | None = None
    minute: int | None = None
    second: int | None = None

    json_string: str | None = None
    data_key: str | None = None
    item_variable: str | None = None
    loop_id: str | None = None
    loop_mode: str | None = None
    range_start: int | float | None = None
    range_end: int | float | None = None
    range_step: int | float | None = None

    headers: dict[str, Any] | None = None
    body: Any = None
    response_variable: str | None = None
    store_headers: bool | None = None

    output: str | None = None
    flow: dict[str, Any] | None = None
    slug: str | None = None
    params: dict[str, Any] | None = None

    key: str | None = None
    uuid: str | None = None
    file: str | None = None
    filetype: str | None = None
    filesize: int | float | None = None

    node_type: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    routes: list[RouteSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_shape(self) -> "StepSpec":
        if not self.id.strip():
            raise ValueError("step id must not be empty")
        if not self.kind.strip():
            raise ValueError("step kind must not be empty")
        return self


class FlowDocument(DSLBaseModel):
    meta: FlowMeta = Field(default_factory=FlowMeta)
    triggers: list[StepSpec] = Field(default_factory=list)
    steps: list[StepSpec] = Field(default_factory=list)
    routes: list[RouteSpec] = Field(default_factory=list)


class DSLDocument(DSLBaseModel):
    flow: FlowDocument = Field(default_factory=FlowDocument)
