package com.atomtag.data

enum class AppMode(val label: String) {
    Sandbox("Sandbox"),
    Teleop("Teleop"),
    SoloSoccer("Solo soccer"),
    TeamSoccer("Team soccer"),
}

enum class DeviceHealth { Ok, Warning, Error }
