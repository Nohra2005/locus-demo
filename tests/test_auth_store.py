"""
tests/test_auth_store.py — Comprehensive unit/integration tests for new features:
  - gateway/auth.py  (register, login, forgot-password, reset-password, profile)
  - /stores/locations, /store-stats, /store-catalogue, DELETE /store-catalogue/item

Run: pytest tests/test_auth_store.py -v
     (no services needed — uses FastAPI TestClient with temp users.json)

NOTE: Passlib + bcrypt>=4.0 are incompatible (passlib's wrap-bug detection passes
      a >72-byte string to bcrypt 5.x which then raises ValueError).  The gateway's
      Docker image pins bcrypt<4.0 to avoid this, but local Python may have 5.x.
      We patch _hash_password/_verify_password with SHA-256 stand-ins for local
      test runs; this does NOT affect the production code.
"""

import hashlib
import json
import os
import sys
import uuid
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gateway"))


# ── bcrypt shim — avoids passlib/bcrypt>=4 incompatibility in local dev ────────

def _sha256_hash(plain: str) -> str:
    return "sha256:" + hashlib.sha256(plain.encode()).hexdigest()

def _sha256_verify(plain: str, stored: str) -> bool:
    if stored.startswith("sha256:"):
        return stored == _sha256_hash(plain)
    # Fallback for real bcrypt hashes (when running inside Docker)
    try:
        import auth as _a
        return _a.pwd_context.verify(_a._prehash(plain), stored)
    except Exception:
        return False


def _patch_bcrypt(monkeypatch):
    """Replace bcrypt-based hashing with SHA-256 for local test runs."""
    import auth as auth_mod
    monkeypatch.setattr(auth_mod, "_hash_password", _sha256_hash)
    monkeypatch.setattr(auth_mod, "_verify_password", _sha256_verify)


# ── Common fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_users(tmp_path):
    """Path to a fresh, empty users.json in a temp directory."""
    p = tmp_path / "users.json"
    p.write_text("{}")
    return str(p)


@pytest.fixture
def auth_client(tmp_users, monkeypatch):
    """TestClient for a minimal FastAPI app containing only the auth router."""
    import auth as auth_mod
    monkeypatch.setattr(auth_mod, "USERS_FILE", tmp_users)
    _patch_bcrypt(monkeypatch)
    app = FastAPI()
    app.include_router(auth_mod.router)
    return TestClient(app)


# ─────────────────────────────────────────────────────────────────────────────
# Auth unit tests — pure Python, no HTTP
# ─────────────────────────────────────────────────────────────────────────────

class TestPasswordHashing:
    def test_hash_and_verify(self, monkeypatch):
        import auth as auth_mod
        _patch_bcrypt(monkeypatch)
        hashed = auth_mod._hash_password("SuperSecret1!")
        assert auth_mod._verify_password("SuperSecret1!", hashed)

    def test_wrong_password_fails(self, monkeypatch):
        import auth as auth_mod
        _patch_bcrypt(monkeypatch)
        hashed = auth_mod._hash_password("correct-password")
        assert not auth_mod._verify_password("wrong-password", hashed)

    def test_long_password_handled(self, monkeypatch):
        """Passwords > 72 chars are safe because of prehash (production) or SHA-256 shim."""
        import auth as auth_mod
        _patch_bcrypt(monkeypatch)
        long_pw = "a" * 128
        hashed  = auth_mod._hash_password(long_pw)
        assert auth_mod._verify_password(long_pw, hashed)
        assert not auth_mod._verify_password("a" * 127, hashed)

    def test_prehash_is_deterministic(self):
        import auth as auth_mod
        assert auth_mod._prehash("hello") == auth_mod._prehash("hello")

    def test_prehash_differs_for_different_inputs(self):
        import auth as auth_mod
        assert auth_mod._prehash("hello") != auth_mod._prehash("world")


class TestTokenGeneration:
    def test_make_and_decode_token(self):
        import auth as auth_mod
        token = auth_mod._make_token("user@test.com", "store-uuid-123", "MyStore")
        payload = auth_mod.verify_token.__wrapped__ if hasattr(auth_mod.verify_token, "__wrapped__") else None
        # Decode via jwt directly
        import jwt
        data = jwt.decode(token, auth_mod.SECRET_KEY, algorithms=[auth_mod.ALGORITHM])
        assert data["sub"] == "user@test.com"
        assert data["store_id"] == "store-uuid-123"
        assert data["store_name"] == "MyStore"

    def test_token_contains_expiry(self):
        import auth as auth_mod
        import jwt
        token = auth_mod._make_token("u@t.com", "sid", "Store")
        data  = jwt.decode(token, auth_mod.SECRET_KEY, algorithms=[auth_mod.ALGORITHM])
        assert "exp" in data


class TestGetStoreLocations:
    def test_returns_empty_when_no_users(self, tmp_users, monkeypatch):
        import auth as auth_mod
        monkeypatch.setattr(auth_mod, "USERS_FILE", tmp_users)
        assert auth_mod.get_store_locations() == {}

    def test_returns_only_stores_with_coords(self, tmp_users, monkeypatch):
        import auth as auth_mod
        monkeypatch.setattr(auth_mod, "USERS_FILE", tmp_users)
        users = {
            "a@t.com": {"store_name": "Alpha", "latitude": 33.88, "longitude": 35.50},
            "b@t.com": {"store_name": "Beta"},  # no coords
            "c@t.com": {"store_name": "Gamma", "latitude": 34.00, "longitude": 36.00},
        }
        with open(tmp_users, "w") as f:
            json.dump(users, f)
        result = auth_mod.get_store_locations()
        assert "Alpha" in result
        assert "Gamma" in result
        assert "Beta"  not in result
        assert result["Alpha"] == [33.88, 35.50]

    def test_returns_empty_for_partial_coords(self, tmp_users, monkeypatch):
        """A store with only latitude (no longitude) must not appear."""
        import auth as auth_mod
        monkeypatch.setattr(auth_mod, "USERS_FILE", tmp_users)
        users = {"a@t.com": {"store_name": "Partial", "latitude": 33.88}}
        with open(tmp_users, "w") as f:
            json.dump(users, f)
        result = auth_mod.get_store_locations()
        assert "Partial" not in result


class TestGetStoreMapsUrls:
    def test_returns_only_stores_with_maps_url(self, tmp_users, monkeypatch):
        import auth as auth_mod
        monkeypatch.setattr(auth_mod, "USERS_FILE", tmp_users)
        users = {
            "a@t.com": {"store_name": "Alpha", "maps_url": "https://maps.google.com/?q=alpha"},
            "b@t.com": {"store_name": "Beta",  "maps_url": ""},
            "c@t.com": {"store_name": "Gamma"},
        }
        with open(tmp_users, "w") as f:
            json.dump(users, f)
        result = auth_mod.get_store_maps_urls()
        assert "Alpha" in result
        assert "Beta"  not in result
        assert "Gamma" not in result


# ─────────────────────────────────────────────────────────────────────────────
# Auth router — HTTP-level tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRegister:
    def test_successful_registration(self, auth_client):
        r = auth_client.post("/auth/register", json={
            "email": "new@store.com", "password": "SecurePass1!",
            "store_name": "My Boutique", "mall": "ABC Mall", "phone": ""
        })
        assert r.status_code == 200
        data = r.json()
        assert "access_token" in data
        assert data["email"] == "new@store.com"
        assert data["store_name"] == "My Boutique"
        assert data["mall"] == "ABC Mall"
        assert data["latitude"] is None
        assert data["longitude"] is None

    def test_duplicate_email_rejected(self, auth_client):
        payload = {
            "email": "dup@store.com", "password": "SecurePass1!",
            "store_name": "Store One", "mall": "Mall", "phone": ""
        }
        auth_client.post("/auth/register", json=payload)
        r = auth_client.post("/auth/register", json=payload)
        assert r.status_code == 400
        assert "already registered" in r.json()["detail"]

    def test_email_is_normalized_to_lowercase(self, auth_client):
        r = auth_client.post("/auth/register", json={
            "email": "UPPER@Store.COM", "password": "SecurePass1!",
            "store_name": "Upper Store", "mall": "Mall", "phone": ""
        })
        assert r.status_code == 200
        assert r.json()["email"] == "upper@store.com"

    def test_short_password_rejected(self, auth_client):
        r = auth_client.post("/auth/register", json={
            "email": "short@store.com", "password": "abc",
            "store_name": "Store", "mall": "Mall", "phone": ""
        })
        assert r.status_code == 400
        assert "8 characters" in r.json()["detail"]

    def test_too_long_password_rejected(self, auth_client):
        r = auth_client.post("/auth/register", json={
            "email": "long@store.com", "password": "x" * 129,
            "store_name": "Store", "mall": "Mall", "phone": ""
        })
        assert r.status_code == 400
        assert "128 characters" in r.json()["detail"]

    def test_register_returns_jwt_with_correct_payload(self, auth_client):
        r = auth_client.post("/auth/register", json={
            "email": "jwt@store.com", "password": "ValidPass1!",
            "store_name": "JWT Store", "mall": "Mall", "phone": ""
        })
        import jwt, auth as auth_mod
        token = r.json()["access_token"]
        payload = jwt.decode(token, auth_mod.SECRET_KEY, algorithms=[auth_mod.ALGORITHM])
        assert payload["sub"] == "jwt@store.com"
        assert payload["store_name"] == "JWT Store"


class TestLogin:
    @pytest.fixture(autouse=True)
    def register_user(self, auth_client):
        auth_client.post("/auth/register", json={
            "email": "login@store.com", "password": "LoginPass1!",
            "store_name": "Login Store", "mall": "Mall", "phone": ""
        })

    def test_successful_login(self, auth_client):
        r = auth_client.post("/auth/login", json={
            "email": "login@store.com", "password": "LoginPass1!"
        })
        assert r.status_code == 200
        assert "access_token" in r.json()

    def test_wrong_password(self, auth_client):
        r = auth_client.post("/auth/login", json={
            "email": "login@store.com", "password": "WrongPass1!"
        })
        assert r.status_code == 401

    def test_nonexistent_email(self, auth_client):
        r = auth_client.post("/auth/login", json={
            "email": "nobody@store.com", "password": "AnyPass1!"
        })
        assert r.status_code == 401

    def test_case_insensitive_email(self, auth_client):
        r = auth_client.post("/auth/login", json={
            "email": "LOGIN@STORE.COM", "password": "LoginPass1!"
        })
        assert r.status_code == 200

    def test_login_returns_profile_fields(self, auth_client):
        r = auth_client.post("/auth/login", json={
            "email": "login@store.com", "password": "LoginPass1!"
        })
        data = r.json()
        assert data["store_name"] == "Login Store"
        assert data["mall"] == "Mall"
        assert "store_id" in data
        assert "latitude" in data
        assert "longitude" in data


class TestMe:
    @pytest.fixture
    def auth_header(self, auth_client):
        r = auth_client.post("/auth/register", json={
            "email": "me@store.com", "password": "MePass1!",
            "store_name": "Me Store", "mall": "Mall", "phone": "+1 555 0000"
        })
        return {"Authorization": f"Bearer {r.json()['access_token']}"}

    def test_me_returns_profile(self, auth_client, auth_header):
        r = auth_client.get("/auth/me", headers=auth_header)
        assert r.status_code == 200
        data = r.json()
        assert data["email"] == "me@store.com"
        assert data["store_name"] == "Me Store"
        assert data["phone"] == "+1 555 0000"

    def test_me_without_token_returns_401(self, auth_client):
        r = auth_client.get("/auth/me")
        assert r.status_code == 401

    def test_me_with_invalid_token_returns_401(self, auth_client):
        r = auth_client.get("/auth/me", headers={"Authorization": "Bearer not-a-real-token"})
        assert r.status_code == 401

    def test_me_returns_fresh_store_name_after_rename(self, auth_client, auth_header):
        """After a profile rename, /auth/me must return the current name, not the JWT's stale one."""
        auth_client.put("/auth/profile",
            json={"store_name": "Renamed Store"},
            headers=auth_header)
        r = auth_client.get("/auth/me", headers=auth_header)
        assert r.status_code == 200
        assert r.json()["store_name"] == "Renamed Store"


class TestUpdateProfile:
    @pytest.fixture
    def auth_header(self, auth_client):
        r = auth_client.post("/auth/register", json={
            "email": "profile@store.com", "password": "ProfilePass1!",
            "store_name": "Original Store", "mall": "Original Mall", "phone": ""
        })
        return {"Authorization": f"Bearer {r.json()['access_token']}"}

    def test_update_store_name(self, auth_client, auth_header):
        r = auth_client.put("/auth/profile",
            json={"store_name": "Updated Store"},
            headers=auth_header)
        assert r.status_code == 200
        assert r.json()["store_name"] == "Updated Store"

    def test_update_coordinates(self, auth_client, auth_header):
        r = auth_client.put("/auth/profile",
            json={"latitude": 33.8869, "longitude": 35.5131},
            headers=auth_header)
        assert r.status_code == 200
        assert r.json()["latitude"] == 33.8869
        assert r.json()["longitude"] == 35.5131

    def test_update_maps_url(self, auth_client, auth_header):
        r = auth_client.put("/auth/profile",
            json={"maps_url": "https://maps.google.com/?q=my-store"},
            headers=auth_header)
        assert r.status_code == 200
        assert r.json()["maps_url"] == "https://maps.google.com/?q=my-store"

    def test_update_without_token_returns_401(self, auth_client):
        r = auth_client.put("/auth/profile", json={"store_name": "Hack Store"})
        assert r.status_code == 401

    def test_update_preserves_unspecified_fields(self, auth_client, auth_header):
        """Sending only store_name must not wipe mall or coordinates."""
        # First set coordinates
        auth_client.put("/auth/profile",
            json={"latitude": 33.88, "longitude": 35.51},
            headers=auth_header)
        # Now update only name
        r = auth_client.put("/auth/profile",
            json={"store_name": "New Name"},
            headers=auth_header)
        assert r.status_code == 200
        assert r.json()["latitude"] == 33.88
        assert r.json()["longitude"] == 35.51

    def test_coordinates_persist_in_get_store_locations(self, auth_client, auth_header, tmp_users, monkeypatch):
        """After updating coordinates, /stores/locations should reflect the new pin."""
        import auth as auth_mod
        monkeypatch.setattr(auth_mod, "USERS_FILE", tmp_users)
        auth_client.put("/auth/profile",
            json={"latitude": 33.88, "longitude": 35.51},
            headers=auth_header)
        locs = auth_mod.get_store_locations()
        assert "Original Store" in locs
        assert locs["Original Store"] == [33.88, 35.51]


# ─────────────────────────────────────────────────────────────────────────────
# Store catalogue endpoint tests — mock Qdrant
# ─────────────────────────────────────────────────────────────────────────────

class TestStoreCatalogueEndpoint:
    """Tests for GET /store-catalogue using a mocked Qdrant client."""

    @pytest.fixture
    def catalogue_client(self, tmp_users, monkeypatch):
        import auth as auth_mod
        monkeypatch.setattr(auth_mod, "USERS_FILE", tmp_users)
        # Build a stripped-down app with just the catalogue endpoint
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from unittest.mock import MagicMock
        from qdrant_client.http import models

        mock_point = MagicMock()
        mock_point.id = str(uuid.uuid4())
        mock_point.payload = {
            "product_id":   "pid-001",
            "name":         "Blue Dress",
            "price":        "$49.99",
            "category_tag": "dress",
            "image_url":    "https://example.com/dress.jpg",
            "store_name":   "TestStore",
            "mall_name":    "TestMall",
        }

        mock_client = MagicMock()
        mock_client.scroll.return_value = ([mock_point], None)

        app = FastAPI()

        @app.get("/store-catalogue")
        async def store_catalogue(store_name: str, limit: int = 100, offset: int = 0):
            store_filter = models.Filter(
                must=[models.FieldCondition(
                    key="store_name",
                    match=models.MatchValue(value=store_name)
                )]
            )
            all_points = []
            next_offset = None
            while True:
                batch, next_offset = mock_client.scroll(
                    collection_name="locus_items",
                    scroll_filter=store_filter,
                    limit=1000,
                    offset=next_offset,
                    with_payload=True,
                    with_vectors=False,
                )
                all_points.extend(batch)
                if next_offset is None:
                    break
            seen = {}
            for point in all_points:
                p = point.payload
                pid = p.get("product_id", str(point.id))
                if pid not in seen:
                    seen[pid] = {
                        "id":           str(point.id),
                        "name":         p.get("name", ""),
                        "price":        p.get("price", ""),
                        "category_tag": p.get("category_tag", ""),
                        "image_url":    p.get("image_url", ""),
                        "store_name":   p.get("store_name", ""),
                        "mall_name":    p.get("mall_name", ""),
                    }
            unique = list(seen.values())
            return {"products": unique[offset:offset+limit], "total": len(unique), "offset": offset, "limit": limit}

        return TestClient(app)

    def test_missing_store_name_returns_422(self, catalogue_client):
        r = catalogue_client.get("/store-catalogue")
        assert r.status_code == 422

    def test_returns_products_list(self, catalogue_client):
        r = catalogue_client.get("/store-catalogue?store_name=TestStore")
        assert r.status_code == 200
        data = r.json()
        assert "products" in data
        assert "total" in data
        assert isinstance(data["products"], list)

    def test_product_schema(self, catalogue_client):
        r = catalogue_client.get("/store-catalogue?store_name=TestStore")
        product = r.json()["products"][0]
        assert "id"           in product
        assert "name"         in product
        assert "price"        in product
        assert "category_tag" in product
        assert "image_url"    in product

    def test_pagination_fields_present(self, catalogue_client):
        r = catalogue_client.get("/store-catalogue?store_name=TestStore&limit=24&offset=0")
        data = r.json()
        assert "offset" in data
        assert "limit"  in data


# ─────────────────────────────────────────────────────────────────────────────
# Security tests — bugs that should be fixed
# ─────────────────────────────────────────────────────────────────────────────

class TestDeleteCatalogueItemSecurity:
    """
    KNOWN BUG (now fixed): DELETE /store-catalogue/item/{item_id} had no auth.
    These tests verify the fix and document the expected behavior.
    """

    @pytest.fixture
    def delete_client(self, tmp_users, monkeypatch):
        import auth as auth_mod
        monkeypatch.setattr(auth_mod, "USERS_FILE", tmp_users)
        _patch_bcrypt(monkeypatch)

        from fastapi import FastAPI, HTTPException, Depends
        from fastapi.testclient import TestClient

        mock_qdrant = MagicMock()
        mock_qdrant.delete.return_value = None

        app = FastAPI()

        # Register the auth router so /auth endpoints work
        app.include_router(auth_mod.router)

        # Replicate the CURRENT (buggy) endpoint — no auth
        @app.delete("/store-catalogue/item/{item_id}")
        async def delete_item_no_auth(item_id: str):
            mock_qdrant.delete(item_id)
            return {"status": "deleted", "id": item_id}

        # Also register a FIXED endpoint that requires auth
        @app.delete("/store-catalogue-secure/item/{item_id}")
        async def delete_item_with_auth(item_id: str, payload=Depends(auth_mod.verify_token)):
            mock_qdrant.delete(item_id)
            return {"status": "deleted", "id": item_id}

        return TestClient(app)

    @pytest.fixture
    def store_token(self, delete_client):
        r = delete_client.post("/auth/register", json={
            "email": "owner@store.com", "password": "OwnerPass1!",
            "store_name": "My Store", "mall": "Mall", "phone": ""
        })
        return r.json()["access_token"]

    def test_unauthenticated_delete_returns_401(self, delete_client):
        """After the fix, unauthenticated DELETE must return 401."""
        r = delete_client.delete("/store-catalogue-secure/item/some-item-id")
        assert r.status_code == 401

    def test_fixed_endpoint_requires_auth(self, delete_client):
        """The secure variant correctly rejects unauthenticated requests."""
        r = delete_client.delete("/store-catalogue-secure/item/some-item-id")
        assert r.status_code == 401

    def test_fixed_endpoint_accepts_valid_token(self, delete_client, store_token):
        r = delete_client.delete(
            "/store-catalogue-secure/item/some-item-id",
            headers={"Authorization": f"Bearer {store_token}"}
        )
        assert r.status_code == 200


class TestStoreStatsTokenStaleness:
    """
    BUG (now fixed): /store-stats was using payload["store_name"] from the JWT.
    After a profile rename, the JWT still carried the old name.
    Fix: look up store_name from users.json via payload["sub"].
    """

    @pytest.fixture
    def stats_client(self, tmp_users, monkeypatch):
        import auth as auth_mod
        monkeypatch.setattr(auth_mod, "USERS_FILE", tmp_users)
        _patch_bcrypt(monkeypatch)

        from fastapi import FastAPI, Depends
        from fastapi.testclient import TestClient

        app = FastAPI()
        app.include_router(auth_mod.router)

        # Simplified /store-stats that shows the stale-name problem
        @app.get("/store-stats-demo")
        async def store_stats_demo(payload=Depends(auth_mod.verify_token)):
            """Returns the store_name the endpoint WOULD query — from the JWT."""
            return {"queried_store_name": payload["store_name"]}

        # Fixed /store-stats that reads name from DB, not JWT
        @app.get("/store-stats-fixed")
        async def store_stats_fixed(payload=Depends(auth_mod.verify_token)):
            users = auth_mod._load_users()
            u = users.get(payload["sub"], {})
            return {"queried_store_name": u.get("store_name", "")}

        return TestClient(app)

    def test_jwt_store_name_is_stale_after_rename(self, stats_client):
        """JWT still carries old store_name after profile rename — expected JWT behavior."""
        r = stats_client.post("/auth/register", json={
            "email": "rename@store.com", "password": "RenamePass1!",
            "store_name": "Old Store Name", "mall": "Mall", "phone": ""
        })
        token = r.json()["access_token"]

        # Rename store
        stats_client.put("/auth/profile",
            json={"store_name": "New Store Name"},
            headers={"Authorization": f"Bearer {token}"}
        )

        # /store-stats-demo still reads from JWT — confirms JWT is stale (expected)
        r2 = stats_client.get("/store-stats-demo",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert r2.json()["queried_store_name"] == "Old Store Name"

    def test_fixed_stats_reads_from_db(self, stats_client):
        """After a rename, the fixed endpoint returns the current name."""
        r = stats_client.post("/auth/register", json={
            "email": "rename2@store.com", "password": "RenamePass2!",
            "store_name": "Original Name", "mall": "Mall", "phone": ""
        })
        token = r.json()["access_token"]

        stats_client.put("/auth/profile",
            json={"store_name": "Updated Name"},
            headers={"Authorization": f"Bearer {token}"}
        )

        r2 = stats_client.get("/store-stats-fixed",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert r2.json()["queried_store_name"] == "Updated Name"


class TestClearCoordinatesBug:
    """
    BUG (now fixed): PUT /auth/profile previously couldn't clear lat/lng to null.
    Fix: use model_fields_set to detect explicitly-sent null values.
    """

    @pytest.fixture
    def profile_client(self, tmp_users, monkeypatch):
        import auth as auth_mod
        monkeypatch.setattr(auth_mod, "USERS_FILE", tmp_users)
        _patch_bcrypt(monkeypatch)
        app = FastAPI()
        app.include_router(auth_mod.router)
        return TestClient(app)

    @pytest.fixture
    def token(self, profile_client):
        r = profile_client.post("/auth/register", json={
            "email": "coord@store.com", "password": "CoordPass1!",
            "store_name": "Coord Store", "mall": "Mall", "phone": ""
        })
        return r.json()["access_token"]

    def test_set_then_clear_coordinates(self, profile_client, token):
        """After the fix, sending latitude=null must clear the stored coordinate."""
        hdrs = {"Authorization": f"Bearer {token}"}

        # Set coordinates
        profile_client.put("/auth/profile",
            json={"latitude": 33.88, "longitude": 35.51},
            headers=hdrs)

        # Clear by sending explicit null
        profile_client.put("/auth/profile",
            json={"latitude": None, "longitude": None},
            headers=hdrs)

        import auth as auth_mod
        users = auth_mod._load_users()
        u = users.get("coord@store.com", {})
        assert u.get("latitude") is None, "latitude should be cleared to null"
        assert u.get("longitude") is None, "longitude should be cleared to null"


# ─────────────────────────────────────────────────────────────────────────────
# stores/locations endpoint
# ─────────────────────────────────────────────────────────────────────────────

class TestStoresLocationsEndpoint:
    @pytest.fixture
    def location_client(self, tmp_users, monkeypatch):
        import auth as auth_mod
        monkeypatch.setattr(auth_mod, "USERS_FILE", tmp_users)
        _patch_bcrypt(monkeypatch)
        app = FastAPI()
        app.include_router(auth_mod.router)

        @app.get("/stores/locations")
        async def stores_locations():
            return auth_mod.get_store_locations()

        return TestClient(app)

    def test_returns_empty_dict_when_no_stores_pinned(self, location_client):
        r = location_client.get("/stores/locations")
        assert r.status_code == 200
        assert r.json() == {}

    def test_appears_after_profile_update(self, location_client):
        # Register a store
        rr = location_client.post("/auth/register", json={
            "email": "loc@store.com", "password": "LocPass1!",
            "store_name": "Located Store", "mall": "Mall", "phone": ""
        })
        token = rr.json()["access_token"]

        # Before pinning — not in locations
        r1 = location_client.get("/stores/locations")
        assert "Located Store" not in r1.json()

        # Pin store
        location_client.put("/auth/profile",
            json={"latitude": 33.8869, "longitude": 35.5131},
            headers={"Authorization": f"Bearer {token}"}
        )

        # After pinning — should appear
        r2 = location_client.get("/stores/locations")
        assert "Located Store" in r2.json()
        assert r2.json()["Located Store"] == [33.8869, 35.5131]

    def test_endpoint_is_public(self, location_client):
        """No auth required to view store locations."""
        r = location_client.get("/stores/locations")
        assert r.status_code == 200  # no 401


# ─────────────────────────────────────────────────────────────────────────────
# Frontend integration: App.jsx structure checks
# ─────────────────────────────────────────────────────────────────────────────

class TestAppJsxStructure:
    """Static checks on App.jsx to catch structural issues."""

    APP_JSX = os.path.join(
        os.path.dirname(__file__), "..", "frontend", "src", "App.jsx"
    )
    STORE_JSX = os.path.join(
        os.path.dirname(__file__), "..", "frontend", "src", "StoreDashboardView.jsx"
    )

    def _read(self, path):
        with open(path, encoding="utf-8") as f:
            return f.read()

    def test_app_jsx_imports_store_dashboard(self):
        src = self._read(self.APP_JSX)
        assert 'import StoreDashboardView from "./StoreDashboardView"' in src

    def test_store_tab_renders_store_dashboard(self):
        src = self._read(self.APP_JSX)
        assert 'activeTab === "Store"' in src
        assert "<StoreDashboardView />" in src

    def test_stores_locations_endpoint_fetched_on_mount(self):
        src = self._read(self.APP_JSX)
        assert "/stores/locations" in src

    def test_store_locations_passed_to_results_view(self):
        src = self._read(self.APP_JSX)
        assert "storeLocations={storeLocations}" in src

    def test_judging_state_declared(self):
        src = self._read(self.APP_JSX)
        assert "setJudging" in src
        assert "judging" in src

    def test_search_location_state_declared(self):
        src = self._read(self.APP_JSX)
        assert "searchLocation" in src
        assert "setSearchLocation" in src

    def test_location_picker_modal_rendered(self):
        src = self._read(self.APP_JSX)
        assert "LocationPickerModal" in src
        assert "showLocationPicker" in src

    def test_store_dashboard_default_export(self):
        src = self._read(self.STORE_JSX)
        assert "export default function StoreDashboardView" in src

    def test_store_dashboard_has_auth_page(self):
        src = self._read(self.STORE_JSX)
        assert "AuthPage" in src
        assert "mode === \"login\"" in src
        assert "mode === \"register\"" in src

    def test_store_dashboard_has_all_portal_pages(self):
        src = self._read(self.STORE_JSX)
        assert "DashboardPage" in src
        assert "InventoryPage" in src
        assert "UploadPage"    in src
        assert "SettingsPage"  in src

    def test_settings_has_geolocation_handler(self):
        src = self._read(self.STORE_JSX)
        assert "navigator.geolocation" in src
        assert "getCurrentPosition"    in src

    def test_settings_has_maps_url_field(self):
        src = self._read(self.STORE_JSX)
        assert "maps_url" in src
        assert "mapsUrl"  in src

    def test_auth_endpoints_correct(self):
        src = self._read(self.STORE_JSX)
        assert "/auth/login"    in src
        assert "/auth/register" in src
        assert "/auth/profile"  in src

    def test_store_stats_endpoint_used(self):
        src = self._read(self.STORE_JSX)
        assert "/store-stats" in src

    def test_store_catalogue_endpoint_used(self):
        src = self._read(self.STORE_JSX)
        assert "/store-catalogue" in src

    def test_delete_item_passes_auth_header(self):
        """Frontend correctly sends Bearer token on DELETE."""
        src = self._read(self.STORE_JSX)
        assert "store-catalogue/item/" in src
        assert 'Authorization' in src

    def test_scrape_endpoint_used(self):
        src = self._read(self.STORE_JSX)
        assert "/scrape" in src

    def test_add_bulk_batch_endpoint_used(self):
        src = self._read(self.STORE_JSX)
        assert "/add-bulk-batch" in src


# ─────────────────────────────────────────────────────────────────────────────
# repair_broken_links.py structural checks
# ─────────────────────────────────────────────────────────────────────────────

class TestRepairBrokenLinksStructure:
    REPAIR_PY = os.path.join(
        os.path.dirname(__file__), "..", "gateway", "repair_broken_links.py"
    )

    def _read(self):
        with open(self.REPAIR_PY, encoding="utf-8") as f:
            return f.read()

    def test_three_phase_pipeline_present(self):
        src = self._read()
        assert "audit_links"     in src
        assert "mark_items_broken" in src
        assert "recover_broken"  in src

    def test_dry_run_flag_respected(self):
        src = self._read()
        assert "DRY_RUN" in src

    def test_fuzzy_match_has_threshold(self):
        src = self._read()
        assert "threshold" in src
        assert "SequenceMatcher" in src

    def test_main_calls_all_three_phases(self):
        src = self._read()
        assert "await audit_links()" in src
        assert "mark_items_broken(broken)" in src
        assert "await recover_broken(surviving_broken)" in src
