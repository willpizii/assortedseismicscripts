import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: asnlib <command> [options]")
        sys.exit(1)

    command = sys.argv[1]
    args = sys.argv[2:]

    if command == "rotate_ccs":
        from asnlib.workflows.rotations import rotate_ccf_folder
        rotate_ccf_folder(args)
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

if __name__ == "__main__":
    main()
