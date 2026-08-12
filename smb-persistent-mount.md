# Persistent SMB Mount on `dgx-1`

This runbook configures the SMB share `//nas.as267.com/llms` to mount at `/mnt/nas`.

> **Security:** The previous SMB password was exposed in terminal/chat output. Change it on the NAS and update the credentials file. Do not put the password directly in `/etc/fstab`.

## 1. Install CIFS support

```bash
sudo apt update
sudo apt install -y cifs-utils
```

## 2. Create the mount point

```bash
sudo mkdir -p /mnt/nas
```

## 3. Configure SMB credentials

Open the protected credentials file:

```bash
sudoedit /etc/.smbcredentials
```

Enter the following, using the newly changed password:

```ini
username=piresbruno
password=NEW_PASSWORD
```

Protect the file so only `root` can read or modify it:

```bash
sudo chown root:root /etc/.smbcredentials
sudo chmod 600 /etc/.smbcredentials
```

## 4. Confirm the local user and group IDs

```bash
id -u piresbruno
id -g piresbruno
```

The configuration below assumes both values are `1000`. Replace `uid=1000` or `gid=1000` if the commands return different values.

## 5. Configure `/etc/fstab`

Edit the file:

```bash
sudoedit /etc/fstab
```

Add or replace the SMB entry with:

```fstab
//nas.as267.com/llms /mnt/nas cifs credentials=/etc/.smbcredentials,vers=3.0,iocharset=utf8,uid=1000,gid=1000,mfsymlinks,_netdev,nofail,x-systemd.automount,x-systemd.idle-timeout=0 0 0
```

Relevant options:

- `credentials=/etc/.smbcredentials` keeps credentials out of `/etc/fstab`.
- `vers=3.0` requests SMB 3.0.
- `uid=1000,gid=1000` makes files appear owned by the local user and group.
- `mfsymlinks` enables Minshall+French symbolic-link support.
- `_netdev` identifies the share as a network filesystem.
- `nofail` prevents an unavailable NAS from blocking system startup.
- `x-systemd.automount` mounts the share when `/mnt/nas` is accessed.
- `x-systemd.idle-timeout=0` prevents systemd from automatically unmounting it due to inactivity.

## 6. Reload and test

```bash
sudo systemctl daemon-reload
sudo mount -a
ls -la /mnt/nas
findmnt /mnt/nas
```

Because this is an automount, accessing `/mnt/nas` with `ls` triggers the actual SMB connection.

## 7. Verify after reboot

```bash
sudo reboot
```

After reconnecting:

```bash
ls -la /mnt/nas
findmnt /mnt/nas
systemctl status mnt-nas.automount --no-pager
systemctl status mnt-nas.mount --no-pager
```

## Troubleshooting

View recent mount-unit logs:

```bash
journalctl -u mnt-nas.automount -u mnt-nas.mount -b --no-pager
```

Test the generated configuration for obvious errors:

```bash
sudo findmnt --verify --verbose
```

If the NAS does not support SMB 3.0, check its supported SMB version before changing the `vers=` option.
