"""
src/extraction/relevance.py
---------------------------
Deterministic relevance filtering for evidence extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.data.join import RequestContext, Message, ImageRef


@dataclass
class RelevantEvidence:
    messages: List[Message]
    images: List[ImageRef]


def get_relevant_evidence(context: RequestContext) -> RelevantEvidence:
    """
    Identifies evidence with a plausible connection to the current request.
    If no relevant evidence is found, returns empty lists to ensure 0 LLM calls.
    """
    relevant_messages: List[Message] = []
    relevant_images: List[ImageRef] = []
    
    seen_msg_ids = set()
    seen_img_ids = set()
    
    def add_msg(msg: Message):
        if msg.message_id not in seen_msg_ids:
            relevant_messages.append(msg)
            seen_msg_ids.add(msg.message_id)
            
    def add_img(img: ImageRef):
        if img.image_id not in seen_img_ids:
            relevant_images.append(img)
            seen_img_ids.add(img.image_id)

    # Priority 1: Unresolved events
    # An unresolved event has amount=NaN or None
    for evt in context.unresolved_events:
        for msg in context.messages_for_events.get(evt.event_id, []):
            add_msg(msg)
        for img in context.images_for_events.get(evt.event_id, []):
            add_img(img)

    # Priority 2: Request-specific messages/images
    for msg in context.messages_for_request:
        add_msg(msg)
    for img in context.images_for_request:
        add_img(img)

    # Priority 3: Event-specific evidence for *resolved* events in the context
    # (Since they are in `context.resolved_events`, they might be amended/cancelled)
    for evt in context.resolved_events:
        for msg in context.messages_for_events.get(evt.event_id, []):
            add_msg(msg)
        for img in context.images_for_events.get(evt.event_id, []):
            add_img(img)

    # Priority 4: User-level messages
    # We do NOT automatically send all 76 user-only messages.
    # We only include a user message if it mentions a keyword related to the request or an event in the context.
    # To keep it deterministic and token-efficient, we only check for event_ids or simple matches.
    # Actually, user-only messages shouldn't be included unless strictly needed.
    # For now, we omit them entirely as per the token optimization rule: 
    # "If there is no deterministic reason they are relevant: do not call the LLM"
    # User-level images are 0 in the dataset anyway.

    return RelevantEvidence(messages=relevant_messages, images=relevant_images)
