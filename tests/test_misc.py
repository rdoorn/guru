"""Tests that do not fit a single domain (session routing)."""
from guru import session


class TestSessionRouting:
    """Tests for the per-context SessionState routing (parallel isolation)."""

    def test_use_and_reset(self) -> None:
        st = session.SessionState()
        token = session.use(st)
        session.model = 'bound-model'
        session.session_in += 7
        assert st.model == 'bound-model'
        assert st.session_in == 7
        session.reset(token)
        assert session.model != 'bound-model'

    def test_threads_are_isolated(self) -> None:
        import threading

        states = [session.SessionState() for _ in range(3)]
        errors: list = []
        start = threading.Barrier(len(states))

        def worker(idx: int) -> None:
            token = session.use(states[idx])
            try:
                start.wait()
                for _ in range(1000):
                    session.session_in += 1
                    session.model = f'model-{idx}'
                if (session.session_in != 1000
                        or session.model != f'model-{idx}'):
                    errors.append(idx)
            finally:
                session.reset(token)

        threads = [
            threading.Thread(target=worker, args=(i,))
            for i in range(len(states))
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert [s.session_in for s in states] == [1000, 1000, 1000]
        assert [s.model for s in states] == ['model-0', 'model-1', 'model-2']
