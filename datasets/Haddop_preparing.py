import os
import re

# ================= SET NORMAL AND ABNORMAL APPLICATION IDS =================
NORMAL = {
    # WordCount Normal
    "application_1445087491445_0005",
    "application_1445087491445_0007",
    "application_1445175094696_0005",
    # PageRank Normal
    "application_1445062781478_0011",
    "application_1445062781478_0016",
    "application_1445062781478_0019",
    "application_1445076437777_0002",
    "application_1445076437777_0005",
    "application_1445144423722_0021",
    "application_1445144423722_0024",
    "application_1445182159119_0012",
}

ABNORMAL = {
    # WordCount Machine down
    "application_1445087491445_0001", "application_1445087491445_0002",
    "application_1445087491445_0003", "application_1445087491445_0004",
    "application_1445087491445_0006", "application_1445087491445_0008",
    "application_1445087491445_0009", "application_1445087491445_0010",
    "application_1445094324383_0001", "application_1445094324383_0002",
    "application_1445094324383_0003", "application_1445094324383_0004",
    "application_1445094324383_0005",
    # WordCount Network disconnection
    "application_1445175094696_0001", "application_1445175094696_0002",
    "application_1445175094696_0003", "application_1445175094696_0004",
    # WordCount Disk full
    "application_1445182159119_0001", "application_1445182159119_0002",
    "application_1445182159119_0003", "application_1445182159119_0004",
    "application_1445182159119_0005",
    # PageRank Machine down
    "application_1445062781478_0012", "application_1445062781478_0013",
    "application_1445062781478_0014", "application_1445062781478_0015",
    "application_1445062781478_0017", "application_1445062781478_0018",
    "application_1445062781478_0020", "application_1445076437777_0001",
    "application_1445076437777_0003", "application_1445076437777_0004",
    "application_1445182159119_0016", "application_1445182159119_0017",
    "application_1445182159119_0018", "application_1445182159119_0019",
    "application_1445182159119_0020",
    # PageRank Network disconnection
    "application_1445144423722_0020", "application_1445144423722_0022",
    "application_1445144423722_0023",
    # PageRank Disk full
    "application_1445182159119_0011", "application_1445182159119_0013",
    "application_1445182159119_0014", "application_1445182159119_0015",
}

# ================= MAIN FUNCTION =================
def main():
    ROOT_FOLDER = "HDO/"  # Update with your path
    APP_PATTERN = re.compile(r"application_\d+_\d+")

    normal_out = open("HDO_normal.log", "w", encoding="utf-8")
    abnormal_out = open("HDO_abnormal.log", "w", encoding="utf-8")

    processed_folders = 0
    skipped_folders = 0
    normal_lines = 0
    abnormal_lines = 0

    for folder_name in os.listdir(ROOT_FOLDER):
        folder_path = os.path.join(ROOT_FOLDER, folder_name)
        if not os.path.isdir(folder_path):
            skipped_folders += 1
            continue

        app_match = APP_PATTERN.match(folder_name)
        if not app_match:
            skipped_folders += 1
            continue

        app_id = app_match.group(0)

        # Decide output file based on exact labels
        if app_id in NORMAL:
            outfile = normal_out
            label_type = "Normal"
        elif app_id in ABNORMAL:
            outfile = abnormal_out
            label_type = "Abnormal"
        else:
            # Ignore unknown applications (strictly follow dataset)
            skipped_folders += 1
            print(f"Skipping unknown application: {app_id}")
            continue

        print(f"Processing {app_id} → {label_type}")
        processed_folders += 1

        # Write all log lines
        for fname in os.listdir(folder_path):
            if not fname.endswith(".log"):
                continue
            fpath = os.path.join(folder_path, fname)
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    outfile.write(line)
                    if label_type == "Normal":
                        normal_lines += 1
                    else:
                        abnormal_lines += 1

    normal_out.close()
    abnormal_out.close()

    print("Finished! Files created:")
    print(" → HDO_normal.log")
    print(" → HDO_abnormal.log")
    print(f"Processed folders: {processed_folders}, Skipped folders: {skipped_folders}")
    print(f"Number of normal rows: {normal_lines}")
    print(f"Number of abnormal rows: {abnormal_lines}")

# ================= ENTRY POINT =================
if __name__ == "__main__":
    main()
