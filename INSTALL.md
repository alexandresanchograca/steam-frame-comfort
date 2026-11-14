# Install Frame Comfort

Needs a Steam Frame with SteamVR. No terminal or root access is needed.

## Install

1. In **Desktop Mode**, download `frame-comfort-VERSION-linux-aarch64.tar.gz`
   from **Releases** and extract it. Keep the extracted folder.
2. In Steam, choose **Add a Non-Steam Game**, browse to that folder and pick
   **`frame-comfort`**. Leave the compatibility tool off.

## First start

1. With SteamVR running, start **frame-comfort** from your library. It
   installs itself and opens the **Frame Comfort** panel in the SteamVR
   dashboard.
2. In the panel, press **RESTART STEAMVR**, then **CONFIRM RESTART**.
   SteamVR restarts once so Comfort can see the game's picture. Do this
   before starting a PC stream, since the restart ends it.

After that, start **frame-comfort** before a game whenever you want Comfort.

## Update

When a new version is out, press **UPDATE** in the panel. If it asks for a
SteamVR restart afterwards, press **RESTART STEAMVR** again.

## Remove

In a terminal, from the extracted folder:

```sh
./frame-comfortctl stop
./frame-comfortctl disable-source --restart --yes
./frame-comfortctl uninstall
```

Your settings stay in `~/.config/vrw-comfort/`. Remove the Steam shortcut
yourself.

## Troubleshooting

- Nothing happens on start: SteamVR must be running first. Details are in
  `~/.local/state/vrw-comfort/frame-comfortctl.log`.
- Game mode shows black: the panel's Mode line says why. A new game needs a
  few seconds of play to be recognised. Plain and Grid always work.
