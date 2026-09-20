from __future__ import annotations

from app.speaker.chain import SpeakerChain
from app.speaker.resolver import RESOLVERS, Resolver
from app.speaker.room import Room
from app.speaker.types import Heard, Speaker

from app.speaker import resolvers as _resolvers  # noqa: F401

RESOLVERS.discover("app.speaker.resolvers")

__all__ = ["RESOLVERS", "Heard", "Resolver", "Room", "Speaker", "SpeakerChain"]
