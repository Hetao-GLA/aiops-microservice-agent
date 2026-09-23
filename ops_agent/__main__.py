"""Run the Agent control plane on the host."""

import uvicorn


def main() -> None:
    uvicorn.run("ops_agent.api:app", host="127.0.0.1", port=8100, reload=False)


if __name__ == "__main__":
    main()
