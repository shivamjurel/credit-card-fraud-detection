"""
Creates the SparkSession used by every pipeline script (and by the backend).

Spark runs on the JVM, so on Windows a few environment settings are needed:
  * JAVA_HOME   -> a Java 17 installation (Spark 3.5 supports Java 8/11/17)
  * HADOOP_HOME -> folder containing bin/winutils.exe + hadoop.dll
                   (needed by Spark to WRITE files such as Parquet and saved models)
  * SPARK_HOME / PYSPARK_PYTHON -> given as short 8.3 paths, because Spark's
                   Windows .cmd scripts break on folder names with spaces
                   ("Credit card fraud detection").

On Linux/macOS/Databricks these fixes are simply skipped.
"""

import os
import sys
from pathlib import Path

import pyspark

HADOOP_HOME_DEFAULT = Path("C:/hadoop")
JAVA17_CANDIDATES = [
    Path("C:/Program Files/Eclipse Adoptium"),
    Path("C:/Program Files/Java"),
    Path("C:/Program Files/Microsoft"),
]


def _short_path(path: Path) -> str:
    """Return the Windows 8.3 short path (no spaces), or the normal path if unavailable."""
    if os.name != "nt":
        return str(path)
    try:
        import ctypes

        buffer = ctypes.create_unicode_buffer(1024)
        if ctypes.windll.kernel32.GetShortPathNameW(str(path), buffer, 1024):
            return buffer.value
    except Exception:
        pass
    return str(path)


def _find_java17() -> str | None:
    for base in JAVA17_CANDIDATES:
        if base.exists():
            for candidate in sorted(base.glob("jdk-17*"), reverse=True):
                if (candidate / "bin" / "java.exe").exists():
                    return str(candidate)
    return None


def configure_windows_environment() -> None:
    """Set the environment variables Spark needs on Windows (no-op elsewhere)."""
    if os.name != "nt":
        return

    java_home = os.environ.get("JAVA_HOME", "")
    if "17" not in java_home:
        java17 = _find_java17()
        if java17:
            os.environ["JAVA_HOME"] = java17

    hadoop_home = Path(os.environ.get("HADOOP_HOME", HADOOP_HOME_DEFAULT))
    if (hadoop_home / "bin" / "winutils.exe").exists():
        os.environ["HADOOP_HOME"] = str(hadoop_home)
        hadoop_bin = str(hadoop_home / "bin")
        if hadoop_bin not in os.environ["PATH"]:
            os.environ["PATH"] = hadoop_bin + os.pathsep + os.environ["PATH"]
    else:
        print(f"[spark] WARNING: winutils.exe not found in {hadoop_home / 'bin'} - saving files will fail.")

    os.environ["SPARK_HOME"] = _short_path(Path(pyspark.__file__).parent)
    python = _short_path(Path(sys.executable))
    os.environ["PYSPARK_PYTHON"] = python
    os.environ["PYSPARK_DRIVER_PYTHON"] = python


def get_spark(app_name: str = "CreditGuard", driver_memory: str = "6g", cores: str = "*"):
    """
    Start (or reuse) a local SparkSession.

    master("local[*]") = run Spark on this computer using all CPU cores. The same
    code runs unchanged on a cluster (YARN, Kubernetes) or Databricks - only the
    master URL changes. That is what makes the pipeline scalable.
    """
    configure_windows_environment()
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.master(f"local[{cores}]")
        .appName(app_name)
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", "48")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark
