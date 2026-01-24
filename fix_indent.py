#!/usr/bin/env python3
# Script to fix indentation in dashboard.py lines 7977-8063

with open('d:\\Updated-FINAL DASH\\dashboard.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Fix lines 7977-8006 (second expander content - add 4 spaces)
for i in range(7976, 8006):  # line numbers are 1-based, so 7977-8006 is indices 7976-8005
    if lines[i].strip():  # If line is not empty
        lines[i] = '    ' + lines[i]

# Fix lines 8008-8037 (third expander - add 4 spaces)  
for i in range(8007, 8037):  # line 8008-8037 is indices 8007-8036
    if lines[i].strip():
        lines[i] = '    ' + lines[i]

# Fix lines 8039-8063 (final markdown - add 4 spaces)
for i in range(8038, 8063):  # line 8039-8063 is indices 8038-8062
    if lines[i].strip():
        lines[i] = '    ' + lines[i]

# Write back
with open('d:\\Updated-FINAL DASH\\dashboard.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Indentation fixed successfully")
