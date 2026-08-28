from lomas_store.migrations import migrate
from lomas_store.repos.answers import AnswerRepo
from lomas_store.repos.classes import ClassRepo
from lomas_store.repos.consent import ConsentRepo
from lomas_store.repos.embeddings import EmbeddingRepo
from lomas_store.repos.events import EventRepo
from lomas_store.repos.orgs import OrgRepo
from lomas_store.repos.schools import SchoolRepo
from lomas_store.repos.sessions import SessionRepo
from lomas_store.repos.students import StudentRepo
from lomas_store.scope import ScopeError, TenantScope
from lomas_store.store import STORES, Store

# Importing the backends module is what registers them.
from lomas_store.backends import sqlite as _sqlite  # noqa: F401

__all__ = [
    "STORES",
    "AnswerRepo",
    "ClassRepo",
    "ConsentRepo",
    "EmbeddingRepo",
    "EventRepo",
    "OrgRepo",
    "SchoolRepo",
    "ScopeError",
    "SessionRepo",
    "Store",
    "StudentRepo",
    "TenantScope",
    "migrate",
]
