from rag.storage.probe import probe_services
from rag.storage.settings import load_settings


def main() -> int:
    settings = load_settings()
    try:
        report = probe_services(settings)
    except Exception as exc:
        print(f"unavailable {exc}")
        return 1
    print(f"milvus connected version={report.milvus_version}")
    print("elasticsearch connected")
    print("object_storage connected")
    print("metadata connected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
