import pytest

from code_agent.llm.types import Message, Request, ToolCall
from code_agent.security.secrets import (
    OutboundRedactor,
    contains_placeholder,
    redact,
    scan,
    shannon_entropy,
)

# Assembled at runtime so this file itself never contains a scannable literal secret.
CLOUD_KEY = "AKIA" + "IOSFODNN7EXAMPLE"
GH = "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"
OPENAI = "sk-proj-" + "Zx9Yw8Vu7Ts6Rq5Po4Nm3Lk2Ji1"
GOOGLE = "AIza" + "SyA1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q"
GOOGLE_NEW = "AQ." + "Zq7Kp2Lm9Xw4Rt6Yv1Bn8Cd3Fg5Hj0_Ek-Ts"
SLACK = "xoxb-" + "123456789012-abcdefABCDEF"
JWT = (
    "eyJhbGciOiJIUzI1NiJ9"
    + ".eyJzdWIiOiIxMjM0NTY3ODkwIn0"
    + ".dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
)
PEM = "-----BEGIN RSA " + "PRIVATE KEY-----\nMIIEpAIBAAKCAQEA\n-----END RSA " + "PRIVATE KEY-----"
RANDOM_B64 = "q8Zt3LmX9pRv2WkY7nBcD4eFgH1jKs0Tu6Vw5Xy+aZ/Q"


@pytest.mark.parametrize(
    ("secret", "kind"),
    [(CLOUD_KEY, "cloud_access_key"), (GH, "github_token"), (OPENAI, "openai_key"),
     (GOOGLE, "google_api_key"), (GOOGLE_NEW, "google_token"), (SLACK, "slack_token"),
     (JWT, "jwt"), (PEM, "private_key"), (RANDOM_B64, "high_entropy")],
)  # fmt: skip
def test_known_formats_are_redacted(secret: str, kind: str):
    text = f"config = '{secret}'  # do not commit"
    clean, kinds = redact(text)
    assert secret not in clean
    assert f"[REDACTED:{kind}]" in clean and kinds == [kind]
    assert clean.endswith("# do not commit")  # surrounding text untouched


def test_url_credentials_and_query_tokens():
    clean, _ = redact("DB = 'postgres://admin:s3cr3tPassw0rd@db:5432/app'")
    assert "s3cr3tPassw0rd" not in clean and "postgres://admin:" in clean and "@db:5432" in clean
    clean, _ = redact("GET https://api.example.com/v1?token=abcd1234efgh5678&page=2")
    assert "abcd1234efgh5678" not in clean and "page=2" in clean


@pytest.mark.parametrize(
    "line",
    ['PASSWORD = "hunter2hunter2"', "api_key: 'k-3xample-value'", 'client_secret="abcdefgh1"'],
)
def test_secret_like_assignments(line: str):
    clean, kinds = redact(line)
    assert kinds == ["assigned_secret"] and "[REDACTED:assigned_secret]" in clean


@pytest.mark.parametrize(
    "safe",
    [
        'digest = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"',  # sha256
        "commit = 'a94a8fe5ccb19ba61c4c0873d391e987982fbbd3'",  # git sha
        "def test_should_do_markup_PY_COLORS_eq_1(monkeypatch): pass",
        "from code_agent.retrieval.search import reciprocal_rank_fusion",
        "see https://mail.python.org/pipermail/pytest-dev/2019-May/004716.html",
        "password = get_password_from_vault()",  # not a literal
        "token = os.environ['GITHUB_TOKEN']",
    ],
)
def test_ordinary_code_is_left_alone(safe: str):
    assert scan(safe) == []


def test_entropy_values():
    assert shannon_entropy("aaaa") == 0
    assert shannon_entropy(RANDOM_B64) > 4.2


def test_placeholders_are_detectable_and_not_re_redacted():
    clean, _ = redact(f"k = '{CLOUD_KEY}'")
    assert contains_placeholder(clean)
    assert redact(clean) == (clean, [])  # idempotent


def test_outbound_redactor_covers_messages_and_tool_arguments():
    redactor = OutboundRedactor()
    request = Request(
        "m",
        (
            Message("system", "rules"),
            Message("user", f"my key is {GH}"),
            Message(
                "assistant", "", (ToolCall("1", "search_codebase", {"query": CLOUD_KEY, "k": 3}),)
            ),
            Message("tool", f"<tool_output>{PEM}</tool_output>", tool_call_id="1"),
        ),
    )
    out = redactor(request)
    flat = repr(out)
    assert GH not in flat and CLOUD_KEY not in flat and "MIIEpAIBAAKCAQEA" not in flat
    assert out.messages[2].tool_calls[0].arguments["k"] == 3  # non-strings untouched
    assert redactor.applied
    assert dict(redactor.redactions) == {"github_token": 1, "cloud_access_key": 1, "private_key": 1}
