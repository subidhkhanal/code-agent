# Records the README demos reproducibly: asciinema captures a real terminal session, expect
# types the user's answers, agg renders the GIF.
#   docker build -f docker/demo.Dockerfile -t code-agent-demo .
#   docker run --rm -t -v "$PWD/docs/demo:/out" code-agent-demo
FROM code-agent:latest
USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends expect curl fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir asciinema \
 && curl -fsSL -o /usr/local/bin/agg \
      https://github.com/asciinema/agg/releases/download/v1.9.0/agg-x86_64-unknown-linux-musl \
 && chmod +x /usr/local/bin/agg
COPY demo /demo
COPY tests/fixtures /demo/fixtures
RUN chown -R agent:agent /demo && chmod +x /demo/record.sh
USER agent
WORKDIR /demo
ENTRYPOINT ["/demo/record.sh"]
