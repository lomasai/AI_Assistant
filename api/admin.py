"""Protected admin APIs for student registration and recognition."""

from __future__ import annotations

import os
import hmac
from typing import Annotated, Any

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from server.face_registration import (
    FaceRegistrationService,
    RecognitionRequest,
    RegistrationError,
    RegistrationSample,
    RegistrationStart,
)
from server.runtime import ApplicationRuntime
from server.student_store import StudentStoreError


class StudentCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    display_name: str = Field(min_length=1, max_length=80)
    grade_level: str | None = Field(default=None, max_length=40)
    language: str | None = Field(default=None, max_length=20)
    consent_given: bool = False


def build_admin_router() -> APIRouter:
    router = APIRouter(prefix="/admin", tags=["admin"])

    def require_admin(x_admin_token: Annotated[str | None, Header()] = None) -> None:
        expected = os.getenv("ADMIN_API_TOKEN", "").strip()
        if expected and not hmac.compare_digest(x_admin_token or "", expected):
            raise HTTPException(status_code=403, detail="Admin authorization required.")

    def runtime_from(request: Request) -> ApplicationRuntime:
        runtime: ApplicationRuntime | None = getattr(request.app.state, "runtime", None)
        if runtime is None:
            raise HTTPException(status_code=503, detail="Runtime is not ready.")
        return runtime

    def registration_service(request: Request) -> FaceRegistrationService:
        runtime = runtime_from(request)
        service: FaceRegistrationService | None = getattr(runtime, "registration", None)
        if service is None:
            raise HTTPException(status_code=503, detail="Registration service is unavailable.")
        return service

    @router.post("/students", summary="Create a student profile")
    async def create_student(
        payload: StudentCreateRequest,
        request: Request,
        x_admin_token: Annotated[str | None, Header()] = None,
    ) -> dict[str, Any]:
        require_admin(x_admin_token)
        runtime = runtime_from(request)
        try:
            student = await runtime.student_store.create_student(
                payload.display_name,
                payload.grade_level,
                payload.language,
                payload.consent_given,
                status="pending",
            )
        except StudentStoreError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"student": FaceRegistrationService._student_view(student)}

    @router.get("/students", summary="List student profiles")
    async def list_students(request: Request, x_admin_token: Annotated[str | None, Header()] = None) -> dict[str, Any]:
        require_admin(x_admin_token)
        runtime = runtime_from(request)
        return {
            "students": [
                FaceRegistrationService._student_view(student)
                for student in await runtime.student_store.list_students()
            ]
        }

    @router.get("/students/{student_id}", summary="Get a student profile")
    async def get_student(
        student_id: str,
        request: Request,
        x_admin_token: Annotated[str | None, Header()] = None,
    ) -> dict[str, Any]:
        require_admin(x_admin_token)
        runtime = runtime_from(request)
        student = await runtime.student_store.get_student(student_id)
        if not student:
            raise HTTPException(status_code=404, detail="Student profile not found.")
        return {"student": FaceRegistrationService._student_view(student)}

    @router.delete("/students/{student_id}", summary="Delete a student profile")
    async def delete_student(
        student_id: str,
        request: Request,
        confirm: bool = False,
        x_admin_token: Annotated[str | None, Header()] = None,
    ) -> dict[str, Any]:
        require_admin(x_admin_token)
        if not confirm:
            raise HTTPException(status_code=422, detail="Profile deletion requires confirm=true.")
        runtime = runtime_from(request)
        deleted = await runtime.student_store.delete_student(student_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Student profile not found.")
        return {"deleted": True}

    @router.post("/registrations", summary="Start student registration")
    async def start_registration(
        payload: RegistrationStart,
        request: Request,
        x_admin_token: Annotated[str | None, Header()] = None,
    ) -> dict[str, Any]:
        require_admin(x_admin_token)
        try:
            return await registration_service(request).start_registration(payload)
        except RegistrationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/registrations/{registration_id}/samples", summary="Submit registration sample")
    async def submit_registration_sample(
        registration_id: str,
        payload: RegistrationSample,
        request: Request,
        x_admin_token: Annotated[str | None, Header()] = None,
    ) -> dict[str, Any]:
        require_admin(x_admin_token)
        try:
            return await registration_service(request).submit_sample(registration_id, payload)
        except RegistrationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/registrations/{registration_id}", summary="Get registration progress")
    async def registration_status(
        registration_id: str,
        request: Request,
        x_admin_token: Annotated[str | None, Header()] = None,
    ) -> dict[str, Any]:
        require_admin(x_admin_token)
        try:
            return await registration_service(request).registration_status(registration_id)
        except RegistrationError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/registrations/{registration_id}/complete", summary="Verify and complete registration")
    async def complete_registration(
        registration_id: str,
        request: Request,
        x_admin_token: Annotated[str | None, Header()] = None,
    ) -> dict[str, Any]:
        require_admin(x_admin_token)
        try:
            return await registration_service(request).complete_registration(registration_id)
        except RegistrationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/registrations/{registration_id}/cancel", summary="Cancel registration")
    async def cancel_registration(
        registration_id: str,
        request: Request,
        x_admin_token: Annotated[str | None, Header()] = None,
    ) -> dict[str, Any]:
        require_admin(x_admin_token)
        try:
            return await registration_service(request).cancel_registration(registration_id)
        except RegistrationError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/recognize", summary="Recognize the current student")
    async def recognize_student(
        payload: RecognitionRequest,
        request: Request,
        x_admin_token: Annotated[str | None, Header()] = None,
    ) -> dict[str, Any]:
        require_admin(x_admin_token)
        return await registration_service(request).recognize_current_student(payload)

    return router


def build_student_identity_router() -> APIRouter:
    router = APIRouter(prefix="/student", tags=["student"])

    @router.get("/profiles", summary="List registered students for manual selection")
    async def list_student_profiles(request: Request) -> dict[str, Any]:
        runtime: ApplicationRuntime | None = getattr(request.app.state, "runtime", None)
        if runtime is None:
            return {"students": []}
        students = await runtime.student_store.list_students()
        return {
            "students": [
                {"id": student["id"], "display_name": student["display_name"]}
                for student in students
                if student.get("registration_status") == "registered"
            ]
        }

    @router.post("/recognize", summary="Recognize the current student for the student UI")
    async def recognize_student(payload: RecognitionRequest, request: Request) -> dict[str, Any]:
        runtime: ApplicationRuntime | None = getattr(request.app.state, "runtime", None)
        if runtime is None or getattr(runtime, "registration", None) is None:
            return {"recognized": False, "display_name": "Guest", "label": "Guest", "student_id": None}
        result = await runtime.registration.recognize_current_student(payload)
        return {
            "recognized": bool(result.get("recognized")),
            "display_name": result.get("display_name") or "Guest",
            "label": result.get("label") or "Guest",
            "student_id": result.get("student_id"),
        }

    return router
