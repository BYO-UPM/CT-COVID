#!/bin/bash

# Read .gitignore file line by line
while IFS= read -r line
do
    # Skip lines that are empty or start with a '#'
    if [[ -z "$line" ]] || [[ $line == \#* ]]; then
        continue
    fi

    # Untrack files in Git that match the pattern
    git ls-files --ignored --exclude-standard | grep -E "$line" | xargs -r git rm --cached
done < .gitignore

# Commit the changes
git commit -m "Untrack files listed in .gitignore"