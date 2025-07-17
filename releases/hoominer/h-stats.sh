#!/bin/bash
# Fetches Hoominer stats and formats for HiveOS

# Source HiveOS environment
[[ -e /hive/bin/hive ]] && . /hive/bin/hive

# Fetch stats from Hoominer API
stats_raw=$(curl -s http://127.0.0.1:8042/gpu)
if [ $? -ne 0 ]; then
    echo "Failed to fetch stats from Hoominer API"
    exit 1
fi

# Check if stats_raw is empty or invalid JSON
if [ -z "$stats_raw" ] || ! echo "$stats_raw" | jq . >/dev/null 2>&1; then
    echo "Invalid or empty API response"
    exit 1
fi

# Parse JSON using jq
khs=$(echo "$stats_raw" | jq '[.hash[] / 1000] | add') # Sum all hashrates in khs
hs=$(echo "$stats_raw" | jq '[.hash[] / 1000]') # Array of hashrates in khs
busid=$(echo "$stats_raw" | jq '.busid')
temp=$(echo "$stats_raw" | jq '.air')
accepted=$(echo "$stats_raw" | jq '.shares.accepted')
rejected=$(echo "$stats_raw" | jq '.shares.rejected')
invalid=$(echo "$stats_raw" | jq '.shares.invalid')
hs_units="khs"
ver=$(echo "$stats_raw" | jq -r '.miner_version')

# Calculate uptime
pid=$(pgrep -f hoominer)
if [ -n "$pid" ]; then
    uptime=$(ps -p "$pid" -o etime= | tr -d '[:space:]')
else
    uptime="0"
fi

# Format stats for HiveOS
stats=$(jq -n \
    --arg khs "$khs" \
    --argjson hs "$hs" \
    --arg hs_units "$hs_units" \
    --argjson temp "$temp" \
    --argjson bus_numbers "$busid" \
    --argjson ar "$(jq -n --argjson accepted "$accepted" --argjson rejected "$rejected" --argjson invalid "$invalid" '[$accepted, $rejected, $invalid]')" \
    --arg uptime "$uptime" \
    --arg ver "$ver" \
    '{khs: $khs, hs: $hs, hs_units: $hs_units, temp: $temp, bus_numbers: $bus_numbers, ar: $ar, uptime: $uptime, ver: $ver}')

echo "$stats"