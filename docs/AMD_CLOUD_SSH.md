# SSH key setup for AMD Developer Cloud

AMD Developer Cloud (https://devcloud.amd.com) requires an SSH **public** key to create GPU droplets.

## Generate a key (if you don't have one)

```powershell
ssh-keygen -t ed25519 -f $env:USERPROFILE\.ssh\id_ed25519_amd -C "amd-devcloud"
```

## Add key to AMD Developer Cloud

1. Log in at https://devcloud.amd.com
2. Go to **Create GPU Droplet** (or account **SSH Keys**)
3. Click **Add an SSH Key**
4. Paste your public key (`id_ed25519_amd.pub` — one line starting with `ssh-ed25519`)
5. Save, then select this key when creating the droplet

Show public key in PowerShell:

```powershell
Get-Content $env:USERPROFILE\.ssh\id_ed25519_amd.pub
```

## Connect to your GPU VM

After the droplet is running, copy its public IP from the dashboard:

```powershell
ssh -i $env:USERPROFILE\.ssh\id_ed25519_amd root@YOUR_DROPLET_IP
```

First connect may ask to trust the host — type `yes`.

## Optional: SSH config shortcut

Add to `$env:USERPROFILE\.ssh\config`:

```
Host amd-gpu
    HostName YOUR_DROPLET_IP
    User root
    IdentityFile ~/.ssh/id_ed25519_amd
```

Then connect with: `ssh amd-gpu`

## Security

- Never commit or share the **private** key
- Destroy GPU droplets when done to avoid burning credits
- See [AMD_FINETUNE_PLAN.md](AMD_FINETUNE_PLAN.md) for GPU workflow
