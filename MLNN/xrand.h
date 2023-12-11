#pragma once
#include <chrono>

auto __xrand_duration = std::chrono::high_resolution_clock::now();
double __xrand_time_offset = 300;

double XRAND2(double input)
{
    double buffer = input;

    for (size_t i = 0; i < 5; i++)
    {
        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = (finish - __xrand_duration);
        buffer += (buffer / 2) + elapsed.count() + __xrand_time_offset;
        __xrand_time_offset += i + 4;
    }

    for (size_t i = 0; i < 30; i++)
    {
        buffer -= floor(pow(10, floor(log10(buffer))));
    }

    return buffer;
}

double XRAND(double upper, double lower)
{
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = (finish - __xrand_duration);
    __xrand_time_offset += 3504789;

    double rand = (double)(XRAND2(((double)elapsed.count() * 1000000000 + __xrand_time_offset) * 1000));
    rand = std::fmod(rand, upper - lower);
    rand += lower;
    return rand;
}