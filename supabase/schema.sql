-- Aro: tabla de partidos con permisos por usuario y link público opcional
create table if not exists public.games (
  id text primary key,
  owner uuid not null default auth.uid() references auth.users (id) on delete cascade,
  data jsonb not null,
  public boolean not null default false,
  updated_at timestamptz not null default now()
);

alter table public.games enable row level security;

grant select on public.games to anon;
grant select, insert, update, delete on public.games to authenticated;

drop policy if exists "games_select" on public.games;
create policy "games_select" on public.games
  for select using (public or owner = auth.uid());

drop policy if exists "games_insert" on public.games;
create policy "games_insert" on public.games
  for insert to authenticated with check (owner = auth.uid());

drop policy if exists "games_update" on public.games;
create policy "games_update" on public.games
  for update to authenticated using (owner = auth.uid()) with check (owner = auth.uid());

drop policy if exists "games_delete" on public.games;
create policy "games_delete" on public.games
  for delete to authenticated using (owner = auth.uid());

create index if not exists games_owner_idx on public.games (owner);
