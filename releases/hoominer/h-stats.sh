#!/bin/bash
# Fetches Hoominer stats and formats for HiveOS

# Fetch stats from Hoominer API
stats_raw=$(curl -s http://127.0.0.1:8042/gpu)
if [ $? -ne 0 ]; then
    echo "Failed to fetch stats from Hoominer API"
    exit 1
fi

# Parse JSON using jq
khs=$(echo "$stats_raw" | jq '.hash[0] / 1000') # Convert hs to khs
hs=($(echo "$stats_raw" | jq -r '.hash[]'))
busid=($(echo "$stats_raw" | jq -r '.busid[]'))
temp=($(echo "$stats_raw" | jq -r '.air[]'))
accepted=($(echo "$stats_raw" | jq -r '.shares.accepted[]'))
rejected=($(echo "$stats_raw" | jq -r '.shares.rejected[]'))
invalid=($(echo "$stats_raw" | jq -r '.shares.invalid[]'))
hs_units="khs"
uptime=$(ps -p $(pgrep -f hoominer) -o etime= | tr -d '[:space:]')
ver=$(echo "$stats_raw" | jq -r '.miner_version')

# Format stats for HiveOS
stats=$(jq -n \
    --arg khs "$khs" \
    --argjson hs "$(echo "${hs[@]}" | jq -c 'map(. / 1000)')" \
    --arg hs_units "$hs_units" \
    --argjson temp "$(echo "${temp[@]}" | jq -c '.')" \
    --argjson busid "$(echo "${busid[@]}" | jq -c '.')" \
    --argjson accepted "$(echo "${accepted[@]}" | jq -c '.')" \
    --argjson rejected "$(echo "${rejected[@]}" | jq -c '.')" \
    --argjson invalid "$(echo "${invalid[@]}" | jq -c '.')" \
    --arg uptime "$uptime" \
    --arg ver "$ver" \
    '{khs: $khs, hs: $hs, hs_units: $hs_units, temp: $temp, bus_numbers: $busid, ar: [$accepted, $rejected, $invalid], uptime: $uptime, ver: $ver}')

echo "$stats"