# Tab completion for the iro command (sourced from ~/.bashrc by tools/bootstrap.sh)
_iro() {
    local cur=${COMP_WORDS[COMP_CWORD]} prev=${COMP_WORDS[1]:-}
    local words
    if [ "$COMP_CWORD" -eq 1 ]; then
        words="help doctor arm pick vision object lidar hw prizm backup up mirror stop topics ros build setup shell code"
    else
        case "$prev" in
            doctor) words="full quick watch jetson devices arm backup --json" ;;
            arm) words="scan calibrate check align zero off" ;;
            object) words="show add train list relabel remove forget memory" ;;
            pick) words="look teach bottle cup --dry-run --drop" ;;
            lidar) words="info reset" ;;
            hw) words="snapshot" ;;
            prizm) words="ping backup restore" ;;
            backup) words="now status log kit verify install" ;;
            up) words="arm lidar camera depth doctor rviz" ;;
        esac
    fi
    COMPREPLY=($(compgen -W "$words" -- "$cur"))
}
complete -F _iro iro
