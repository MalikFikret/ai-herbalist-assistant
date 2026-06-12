import subprocess

def apply_patch():
    try:
        import patch
    except ImportError:
        subprocess.check_call(["pip", "install", "patch"])
        import patch

    pset = patch.fromfile("admin_clean.patch")
    if pset:
        pset.apply()
        print("Patch applied successfully")
    else:
        print("Failed to parse patch")

if __name__ == "__main__":
    apply_patch()
