"""Allow `python -m hypolens.web` to launch the web server."""
import os
from hypolens.web.app import app, _parse_args

if __name__ == "__main__":
    args = _parse_args()
    # Promote CLI LLM args into env vars so /api/summarize picks them up.
    if args.api_key:  os.environ["LLM_API_KEY"]  = args.api_key
    if args.base_url: os.environ["LLM_BASE_URL"] = args.base_url
    if args.model:    os.environ["LLM_MODEL"]    = args.model
    app.config.update(
        DEFAULT_DATASET = args.dataset,
    )
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
