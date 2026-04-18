#!/bin/bash
set -e

start() {
    setsid -f bash -lc '
        source .venv/bin/activate
        exec torchrun --standalone --nproc_per_node=8 nanogpt/gpt_2.py \
        > gpt2.log 2>&1 < /dev/null'
    echo "Started"
}

stop() {
    pgrep -f 'torchrun.*nanogpt/gpt_2.py|nanogpt/gpt_2.py' | xargs -r kill
    echo "Stopped"
}

watch() {
    tail -f -n 200 gpt2.log
}

if [ "$1" == "start" ]; then
    start
elif [ "$1" == "stop" ]; then
    stop
elif [ "$1" == "watch" ]; then
    watch
else
    echo "Usage: ./train.sh <start|stop|watch>"
fi
