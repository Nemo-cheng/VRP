FROM gcc:13-bookworm AS build

COPY src /src

RUN g++ -std=c++11 -O3 \
    -o /evrp-tw-spd \
    /src/evrp_tw_spd_solver.cpp \
    /src/eval.cpp \
    /src/operator.cpp \
    /src/search_framework.cpp \
    /src/solution.cpp \
    /src/util.cpp \
    /src/data.cpp \
    /src/evolution.cpp

FROM debian:bookworm-slim

COPY --from=build /evrp-tw-spd /usr/local/bin/evrp-tw-spd

ENTRYPOINT ["/usr/local/bin/evrp-tw-spd"]
