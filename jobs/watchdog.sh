#!/bin/bash
# Cancels any of the given jobs that are still queued 2 h (MAX_WAIT seconds) after this starts,
# and prints one line per state change. Run on the frontend in the background (needs a ticket).
# Usage: jobs/watchdog.sh JOBID...
MAX_WAIT=${MAX_WAIT:-7200}
# Each nympha login has its own /tmp/krb5cc_<uid>_* cache, removed when that login ends;
# switch to the newest valid one if ours disappears.
fix_krb() {
    klist -s 2>/dev/null && return
    for c in $(ls -t /tmp/krb5cc_$(id -u)_* 2>/dev/null); do
        if klist -s -c "FILE:$c"; then export KRB5CCNAME=FILE:$c; return; fi
    done
}
declare -A last
while :; do
    fix_krb
    active=0
    for j in "$@"; do
        info=$(qstat -xf "$j" 2>/dev/null)
        s=$(awk -F' = ' '/job_state/{print $2}' <<<"$info" | tr -d ' ')
        t0=$(date -d "$(awk -F' = ' '/ qtime/{print $2}' <<<"$info")" +%s 2>/dev/null || date +%s)
        [ "$s" != "${last[$j]}" ] && { echo "$(date +%H:%M) $j ${last[$j]:-?} -> $s"; last[$j]=$s; }
        case $s in
            Q|H) active=1
                 if (( $(date +%s) - t0 > MAX_WAIT )); then
                     qdel "$j" && echo "$(date +%H:%M) $j cancelled: queued > $((MAX_WAIT / 60)) min"
                 fi ;;
            R|E|B) active=1 ;;
            "") active=1 ;;   # qstat failed (e.g. expired Kerberos ticket): keep watching
        esac
    done
    (( active )) || break
    sleep 60
done
for j in "$@"; do
    qstat -xf "$j" 2>/dev/null | awk -v j="$j" -F' = ' '/Exit_status|resources_used.walltime|exec_host /{printf "%s %s=%s\n", j, $1, $2}' | sed 's/^ *//; s/  */ /g'
done
