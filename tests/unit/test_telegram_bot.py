"""Tests for the TelegramCommandBot."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from src.ops.telegram_bot import TelegramCommandBot, _escape_markdown

UTC = ZoneInfo("UTC")


def _sample_state():
    """Build a realistic system state dict for testing."""
    return {
        "account": {
            "equity": 100572.50,
            "balance": 100072.00,
            "unrealized_pnl": 500.50,
        },
        "protocol": {
            "day": 7,
            "duration": 30,
            "status": "running",
            "drawdown": 0.005,
            "degraded_reason": None,
        },
        "positions": [
            {
                "pair": "USD/JPY",
                "units": 5000,
                "entry_price": 152.500,
                "current_price": 153.100,
                "unrealized_pnl": 300.00,
                "stop_price": 149.500,
                "high_water_mark": 153.200,
                "entry_time": datetime(2026, 2, 3, 10, 0, tzinfo=UTC),
                "tranche_count": 2,
            },
            {
                "pair": "AUD/JPY",
                "units": 10330,
                "entry_price": 97.800,
                "current_price": 98.000,
                "unrealized_pnl": 200.50,
                "stop_price": 95.100,
                "high_water_mark": 98.100,
                "entry_time": datetime(2026, 2, 2, 14, 0, tzinfo=UTC),
                "tranche_count": 3,
            },
        ],
        "performance": {
            "daily_pnl": 125.30,
            "high_water_mark": 100600.00,
        },
        "last_tick": datetime(2026, 2, 8, 15, 0, tzinfo=UTC),
        "market_open": True,
    }


@pytest.fixture
def bot():
    """Create a TelegramCommandBot with a mock state provider."""
    return TelegramCommandBot(
        bot_token="test-token-123",
        chat_id="99999",
        state_provider=_sample_state,
    )


class TestTelegramCommandBot:
    """Tests for TelegramCommandBot command handling."""

    def test_handle_status_contains_account_info(self, bot):
        """Status response should include equity, balance, unrealized."""
        state = _sample_state()
        result = bot._handle_status(state)
        assert "100,572.50" in result
        assert "100,072.00" in result
        assert "+500.50" in result

    def test_handle_status_contains_protocol_info(self, bot):
        """Status response should include protocol day and status."""
        state = _sample_state()
        result = bot._handle_status(state)
        assert "7/30" in result
        assert "RUNNING" in result

    def test_handle_status_contains_positions(self, bot):
        """Status response should include all position details."""
        state = _sample_state()
        result = bot._handle_status(state)
        assert "USD/JPY" in result
        assert "AUD/JPY" in result
        assert "5,000" in result
        assert "+300.00" in result

    def test_handle_status_shows_stop_prices(self, bot):
        """Status should show stop-loss levels for each position."""
        state = _sample_state()
        result = bot._handle_status(state)
        assert "149.500" in result
        assert "95.100" in result

    def test_handle_status_degraded_shows_reason(self, bot):
        """Status should show degradation reason when DEGRADED."""
        state = _sample_state()
        state["protocol"]["status"] = "degraded"
        state["protocol"]["degraded_reason"] = "3 consecutive losing days"
        result = bot._handle_status(state)
        assert "DEGRADED" in result
        assert "3 consecutive losing days" in result

    def test_handle_status_no_positions(self, bot):
        """Status should handle empty positions gracefully."""
        state = _sample_state()
        state["positions"] = []
        result = bot._handle_status(state)
        assert "Positions (0)" in result
        assert "$+0.00" in result or "$0.00" in result

    def test_handle_health_connected(self, bot):
        """Health check should show connected status."""
        state = _sample_state()
        result = bot._handle_health(state)
        assert "Connected" in result
        assert "RUNNING" in result
        assert "OPEN" in result

    def test_handle_health_disconnected(self, bot):
        """Health check should show disconnected when equity is 0."""
        state = _sample_state()
        state["account"]["equity"] = 0
        result = bot._handle_health(state)
        assert "DISCONNECTED" in result

    def test_handle_health_market_closed(self, bot):
        """Health check should show market status."""
        state = _sample_state()
        state["market_open"] = False
        result = bot._handle_health(state)
        assert "CLOSED" in result

    def test_handle_health_last_tick_ago(self, bot):
        """Health check should show time since last tick."""
        state = _sample_state()
        state["last_tick"] = datetime.now(UTC) - timedelta(minutes=42)
        result = bot._handle_health(state)
        assert "42m ago" in result

    def test_handle_health_no_last_tick(self, bot):
        """Health check should handle missing last tick."""
        state = _sample_state()
        state["last_tick"] = None
        result = bot._handle_health(state)
        assert "unknown" in result

    def test_handle_positions_detail(self, bot):
        """Positions command should show per-pair detail."""
        state = _sample_state()
        result = bot._handle_positions(state)
        assert "USD/JPY" in result
        assert "AUD/JPY" in result
        assert "tranches: 2" in result
        assert "tranches: 3" in result
        assert "Total PnL" in result

    def test_handle_positions_empty(self, bot):
        """Positions command should handle no positions."""
        state = _sample_state()
        state["positions"] = []
        result = bot._handle_positions(state)
        assert "No open positions" in result

    def test_handle_positions_shows_risk(self, bot):
        """Positions command should show risk from stop."""
        state = _sample_state()
        result = bot._handle_positions(state)
        # USD/JPY: entry 152.5, stop 149.5 → risk ~1.97%
        assert "risk:" in result

    def test_handle_help(self, bot):
        """Help command should list all commands."""
        state = _sample_state()
        result = bot._handle_help(state)
        assert "/status" in result
        assert "/health" in result
        assert "/positions" in result
        assert "/help" in result

    def test_escape_markdown(self):
        """Markdown escaping should handle underscores and backticks."""
        text = "USD_JPY price `152.5` [link]"
        escaped = _escape_markdown(text)
        assert "\\_" in escaped
        assert "\\`" in escaped
        assert "\\[" in escaped

    def test_escape_markdown_preserves_asterisks(self):
        """Markdown escaping should leave * alone for bold."""
        text = "*bold text*"
        escaped = _escape_markdown(text)
        assert escaped == "*bold text*"


class TestBotSecurity:
    """Tests for security: only authorized chat gets responses."""

    def test_ignores_unauthorized_chat(self, bot):
        """Bot should ignore messages from wrong chat_id."""
        update = {
            "update_id": 1,
            "message": {
                "chat": {"id": 11111},  # Wrong chat
                "text": "/status",
            },
        }
        # Should not raise, should not call state_provider
        with patch.object(bot, "_send_message") as mock_send:
            bot._process_update(update)
            mock_send.assert_not_called()

    def test_responds_to_authorized_chat(self, bot):
        """Bot should respond to messages from correct chat_id."""
        update = {
            "update_id": 2,
            "message": {
                "chat": {"id": 99999},  # Correct chat
                "text": "/help",
            },
        }
        with patch.object(bot, "_send_message") as mock_send:
            bot._process_update(update)
            mock_send.assert_called_once()
            assert "/status" in mock_send.call_args[0][0]

    def test_ignores_non_command_messages(self, bot):
        """Bot should ignore messages that don't start with /."""
        update = {
            "update_id": 3,
            "message": {
                "chat": {"id": 99999},
                "text": "hello there",
            },
        }
        with patch.object(bot, "_send_message") as mock_send:
            bot._process_update(update)
            mock_send.assert_not_called()

    def test_handles_unknown_command(self, bot):
        """Bot should reply with error for unknown commands."""
        update = {
            "update_id": 4,
            "message": {
                "chat": {"id": 99999},
                "text": "/foobar",
            },
        }
        with patch.object(bot, "_send_message") as mock_send:
            bot._process_update(update)
            mock_send.assert_called_once()
            assert "Unknown command" in mock_send.call_args[0][0]

    def test_handles_command_with_bot_suffix(self, bot):
        """Bot should strip @bot_name suffix from commands."""
        update = {
            "update_id": 5,
            "message": {
                "chat": {"id": 99999},
                "text": "/status@carry_trade_bot",
            },
        }
        with patch.object(bot, "_send_message") as mock_send:
            bot._process_update(update)
            mock_send.assert_called_once()
            assert "CARRY TRADE STATUS" in mock_send.call_args[0][0]


class TestBotLifecycle:
    """Tests for start/stop lifecycle."""

    def test_start_creates_thread(self, bot):
        """Starting the bot should create a daemon thread."""
        with patch.object(bot, "_register_commands"):
            with patch.object(bot, "_poll_loop"):
                bot.start()
                assert bot._running is True
                assert bot._thread is not None
                assert bot._thread.daemon is True
                bot.stop()

    def test_stop_sets_running_false(self, bot):
        """Stopping the bot should set _running to False."""
        bot._running = True
        bot._thread = MagicMock()
        bot._thread.join = MagicMock()
        bot.stop()
        assert bot._running is False

    def test_double_start_is_noop(self, bot):
        """Starting an already-running bot should be a no-op."""
        bot._running = True
        bot.start()
        # Should not create a new thread
        assert bot._thread is None  # Was never set by second start

    def test_update_id_tracking(self, bot):
        """Bot should track last update_id to avoid processing duplicates."""
        update = {
            "update_id": 42,
            "message": {
                "chat": {"id": 99999},
                "text": "/help",
            },
        }
        with patch.object(bot, "_send_message"):
            bot._process_update(update)
            assert bot._last_update_id == 42


class TestBotAPIInteraction:
    """Tests for Telegram API interaction."""

    def test_send_message_truncates_long_text(self, bot):
        """Messages longer than 4096 chars should be truncated."""
        long_text = "x" * 5000
        with patch.object(bot, "_api_call", return_value={"ok": True}) as mock_api:
            bot._send_message(long_text)
            sent_text = mock_api.call_args[1].get("payload", mock_api.call_args[0][1])
            # The payload text should be truncated
            text_sent = sent_text["text"]
            assert len(text_sent) <= 4096
            assert "truncated" in text_sent

    def test_send_message_passes_chat_id(self, bot):
        """Send message should use the configured chat_id."""
        with patch.object(bot, "_api_call", return_value={"ok": True}) as mock_api:
            bot._send_message("test")
            payload = mock_api.call_args[0][1]
            assert payload["chat_id"] == "99999"

    def test_get_updates_passes_offset(self, bot):
        """getUpdates should use last_update_id + 1 as offset."""
        bot._last_update_id = 100
        with patch.object(
            bot, "_api_call", return_value={"ok": True, "result": []}
        ) as mock_api:
            bot._get_updates()
            params = mock_api.call_args[0][1]
            assert params["offset"] == 101

    def test_get_updates_handles_failure(self, bot):
        """getUpdates should return empty list on failure."""
        with patch.object(bot, "_api_call", return_value=None):
            result = bot._get_updates()
            assert result == []

    def test_state_provider_error_sends_error_message(self, bot):
        """If state_provider raises, bot should send error message."""
        failing_bot = TelegramCommandBot(
            bot_token="test",
            chat_id="99999",
            state_provider=lambda: (_ for _ in ()).throw(RuntimeError("DB locked")),
        )
        update = {
            "update_id": 10,
            "message": {
                "chat": {"id": 99999},
                "text": "/status",
            },
        }
        with patch.object(failing_bot, "_send_message") as mock_send:
            failing_bot._process_update(update)
            mock_send.assert_called_once()
            assert "Error" in mock_send.call_args[0][0]
