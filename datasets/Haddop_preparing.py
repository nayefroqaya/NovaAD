import os
import re

# ================= LABELS =================
NORMAL = {
    "application_1445087491445_0005",
    "application_1445087491445_0007",
    "application_1445175094696_0005",
    "application_1445062781478_0011",
    "application_1445062781478_0016",
    "application_1445062781478_0019",
    "application_1445076437777_0002",
    "application_1445076437777_0005",
    "application_1445144423722_0021",
    "application_1445144423722_0024",
    "application_1445182159119_0012",
}

# All abnormal = Machine Down + Network + Disk Full
ABNORMAL = {
    # Machine Down
    "application_1445087491445_0001",
    "application_1445087491445_0002",
    "application_1445087491445_0003",
    "application_1445087491445_0004",
    "application_1445087491445_0006",
    "application_1445087491445_0008",
    "application_1445087491445_0009",
    "application_1445087491445_0010",
    "application_1445094324383_0001",
    "application_1445094324383_0002",
    "application_1445094324383_0003",
    "application_1445094324383_0004",
    "application_1445094324383_0005",
    "application_1445062781478_0012",
    "application_1445062781478_0013",
    "application_1445062781478_0014",
    "application_1445062781478_0015",
    "application_1445062781478_0017",
    "application_1445062781478_0018",
    "application_1445062781478_0020",
    "application_1445076437777_0001",
    "application_1445076437777_0003",
    "application_1445076437777_0004",
    "application_1445182159119_0016",
    "application_1445182159119_0017",
    "application_1445182159119_0018",
    "application_1445182159119_0019",
    "application_1445182159119_0020",
    # Network Disconnection
    "application_1445175094696_0001",
    "application_1445175094696_0002",
    "application_1445175094696_0003",
    "application_1445175094696_0004",
    "application_1445144423722_0020",
    "application_1445144423722_0022",
    "application_1445144423722_0023",
    # Disk Full
    "application_1445182159119_0001",
    "application_1445182159119_0002",
    "application_1445182159119_0003",
    "application_1445182159119_0004",
    "application_1445182159119_0005",
    "application_1445182159119_0011",
    "application_1445182159119_0013",
    "application_1445182159119_0014",
    "application_1445182159119_0015",
}

# ================= PATH =================
ROOT_FOLDER = "/datasets/Hadoop"  # <- change this to your dataset path

# Regex to extract application IDs from folder names
APP_PATTERN = re.compile(r"application_\d+_\d+")

# Output files
normal_out = open("normal.log", "w", encoding="utf-8")
abnormal_out = open("abnormal.log", "w", encoding="utf-8")
# ================= WALK THROUGH DATASET =================
for folder_name in os.listdir(ROOT_FOLDER):
    folder_path = os.path.join(ROOT_FOLDER, folder_name)

    # Skip non-folders
    if not os.path.isdir(folder_path):
        continue

    # Match application ID
    app_match = APP_PATTERN.match(folder_name)
    if not app_match:
        continue

    app_id = app_match.group(0)

    # Decide if normal or abnormal
    if app_id in NORMAL:
        outfile = normal_out
    elif app_id in ABNORMAL:
        outfile = abnormal_out
    else:
        # Skip unknown apps
        continue

    print(f"Processing {app_id} → {'Normal' if app_id in NORMAL else 'Abnormal'}")

    # Read all .log files in this folder
    for fname in os.listdir(folder_path):
        if not fname.endswith(".log"):
            continue
        fpath = os.path.join(folder_path, fname)
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                outfile.write(line)
# ================= CLOSE FILES =================
normal_out.close()
abnormal_out.close()

print("Finished! Files created:")
print(" → normal.log")
print(" → abnormal.log")
