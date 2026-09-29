# The official SWE-bench harness, run on Linux. On a Windows host the harness writes its eval
# scripts with CRLF line endings, which breaks them inside the task containers, so it runs here
# instead, driving the host's Docker daemon through the mounted socket.
#   docker build -f docker/harness.Dockerfile -t swebench-harness:5.0.2 .
FROM python:3.12-slim
RUN pip install --no-cache-dir "swebench==5.0.2"
WORKDIR /run-dir
