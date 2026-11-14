#!/bin/sh
# Start Frame Comfort: set it up or update it from this folder if needed,
# then run it. Add this file to Steam as a Non-Steam game.
exec "$(dirname "$(readlink -f "$0")")/frame-comfortctl" launch
