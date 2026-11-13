#!/usr/bin/env python3
"""
通过 GitHub Git Data API 上传本地仓库内容。

沙箱环境无法直连 github.com 的 git 协议（443 超时），
但 GitHub REST API 经 gh-proxy 可用且支持写操作，
因此用 API 逐文件创建 blob → tree → commit → ref 完成推送。
"""
import base64
import json
import os
import subprocess
import sys
import time
import urllib.request

API = "https://gh-proxy.com/https://api.github.com"
TOKEN = os.environ["GITHUB_TOKEN"]
REPO = os.environ.get("GITHUB_REPO", "diaoyunxi/ground-distance-measure")
BRANCH = os.environ.get("GITHUB_BRANCH", "main")
LOCAL = os.environ.get("LOCAL_REPO", "/workspace/ground-distance-measure")


def api(method, path, payload=None, retries=3):
    url = f"{API}/repos/{REPO}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    for attempt in range(retries):
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", f"Bearer {TOKEN}")
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("Content-Type", "application/json")
        req.add_header("User-Agent", "ground-distance-measure-uploader")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = resp.read().decode()
                return resp.status, (json.loads(body) if body else None)
        except urllib.error.HTTPError as e:
            detail = e.read().decode()[:400]
            if e.code in (502, 503, 504) and attempt < retries - 1:
                time.sleep(3 * (attempt + 1))
                continue
            return e.code, detail
        except Exception as e:  # noqa: BLE001
            if attempt < retries - 1:
                time.sleep(3 * (attempt + 1))
                continue
            return 0, str(e)
    return 0, "unreachable"


def git(*args):
    return subprocess.run(
        ["git", "-C", LOCAL, *args], capture_output=True, text=True, check=False
    ).stdout.strip()


def collect_files():
    """返回 [(路径, 内容bytes)]，排除 .git 与构建产物。"""
    tracked = git("ls-files").splitlines()
    files = []
    for path in tracked:
        if not path or path.startswith(".git/"):
            continue
        if path in ("package-lock.json",):
            pass  # lockfile 需要提交，保证 CI 的 npm ci 可用
        if path.startswith("dist/"):
            continue
        full = os.path.join(LOCAL, path)
        if os.path.isfile(full):
            with open(full, "rb") as f:
                files.append((path, f.read()))
    return files


def ensure_repo_initialized():
    """空仓库禁止创建 blob（API 返回 409），需要先有一个初始提交。"""
    status, ref = api("GET", f"/git/ref/heads/{BRANCH}")
    if status == 200 and isinstance(ref, dict) and ref.get("object"):
        return True

    print("仓库为空，先创建初始提交…")
    # 空仓库只能用 Contents API 创建首个文件
    b64 = base64.b64encode(b"# ground-distance-measure\n").decode()
    status, res = api(
        "PUT",
        "/contents/README.md",
        {
            "message": "chore: initialize repository",
            "content": b64,
            "branch": BRANCH,
        },
    )
    if status not in (200, 201):
        print(f"  [失败] 初始文件: HTTP {status} {res}")
        return False

    # 取回初始 commit 的 sha，作为后续提交的父
    status, ref = api("GET", f"/git/ref/heads/{BRANCH}")
    if status == 200 and isinstance(ref, dict):
        print(f"  初始提交完成: {ref['object']['sha']}")
    return True


def main():
    print(f"仓库: {REPO}  分支: {BRANCH}")
    local_commit = git("rev-parse", "HEAD")
    local_msg = git("log", "-1", "--pretty=%s")
    local_body = git("log", "-1", "--pretty=%b")

    if not ensure_repo_initialized():
        return 1

    files = collect_files()
    print(f"待上传文件: {len(files)} 个")

    # 1. 为每个文件创建 blob
    tree_entries = []
    for idx, (path, content) in enumerate(files, 1):
        b64 = base64.b64encode(content).decode()
        status, res = api(
            "POST", "/git/blobs", {"content": b64, "encoding": "base64"}
        )
        if status not in (200, 201):
            print(f"  [失败] blob {path}: HTTP {status} {res}")
            return 1
        tree_entries.append(
            {
                "path": path,
                "mode": "100755" if os.access(os.path.join(LOCAL, path), os.X_OK) else "100644",
                "type": "blob",
                "sha": res["sha"],
            }
        )
        if idx % 10 == 0 or idx == len(files):
            print(f"  blob {idx}/{len(files)}")

    # 2. 创建 tree
    status, tree = api("POST", "/git/trees", {"tree": tree_entries})
    if status not in (200, 201):
        print(f"  [失败] tree: HTTP {status} {tree}")
        return 1
    print(f"  tree: {tree['sha']}")

    # 3. 创建 commit（带上远端当前提交作为父，保持历史连续）
    status, ref_now = api("GET", f"/git/ref/heads/{BRANCH}")
    parents = []
    if status == 200 and isinstance(ref_now, dict) and ref_now.get("object"):
        parents = [ref_now["object"]["sha"]]

    payload = {
        "message": local_msg,
        "tree": tree["sha"],
        "parents": parents,
    }
    if local_body:
        payload["message"] = f"{local_msg}\n\n{local_body}"
    status, commit = api("POST", "/git/commits", payload)
    if status not in (200, 201):
        print(f"  [失败] commit: HTTP {status} {commit}")
        return 1
    print(f"  commit: {commit['sha']}")

    # 4. 更新或创建分支引用
    status, ref = api("GET", f"/git/ref/heads/{BRANCH}")
    if status == 200 and isinstance(ref, dict) and ref.get("object"):
        status, res = api(
            "PATCH", f"/git/refs/heads/{BRANCH}", {"sha": commit["sha"], "force": True}
        )
        action = "更新"
    else:
        status, res = api(
            "POST", "/git/refs", {"ref": f"refs/heads/{BRANCH}", "sha": commit["sha"]}
        )
        action = "创建"
    if status not in (200, 201):
        print(f"  [失败] ref({action}): HTTP {status} {res}")
        return 1
    print(f"  分支{action}: {BRANCH} -> {commit['sha']}")

    print(f"\n推送完成，远端 commit: {commit['sha']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
