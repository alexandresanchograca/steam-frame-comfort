# Install Frame Comfort

Needs a Steam Frame with SteamVR. No terminal or root access is needed to
install or use it.

## Install

1. In **Desktop Mode**, download `frame-comfort-VERSION-linux-aarch64.tar.gz`
   from **Releases**.
2. Extract it. If right-clicking the download offers no **Extract** option,
   install **Ark** from the **Discover** app first, then open the download
   with Ark and extract it.
3. In the extracted folder, right-click **`frame-comfort`** and choose
   **Add to Steam**. It appears in your library as **frame-comfort**.

## Use

1. Back in Gaming Mode, start **frame-comfort** from your library before a
   game. It installs Comfort the first time, then opens the **Frame
   Comfort** panel in the SteamVR dashboard.
2. The first time, the panel says **GAME MODE NEEDS ONE STEAMVR RESTART**.
   Press **RESTART STEAMVR**, then **CONFIRM RESTART**. The restart ends a PC
   stream, so do it before starting one.

Comfort keeps running across games until you quit it or SteamVR restarts,
so you start it once per SteamVR session.

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
- **Add to Steam** is missing from the right-click menu: the file must be
  executable. Extracting with Ark keeps that; otherwise tick **Is
  executable** under **Properties → Permissions**.
- Game mode shows black: the panel's Mode line says why. A new game needs a
  few seconds of play to be recognised. Plain and Grid always work.
