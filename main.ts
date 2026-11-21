// Sokoban solver for make.fun's sokoban-solver-01: push-optimal A* over pushes, room by room.
//
// A level is a hallway with one to four rooms off it; a box pushed out of its room can never come back, so the
// level's push optimum is the sum of its rooms' optima. Each room is solved in turn with A* over pushes (not
// moves): a state is the room's boxes plus the player's reachable region, named by its lowest square. The
// estimate is a min-cost matching of boxes to goals over single-box push distances (reverse-pull BFS), which
// never overstates the pushes still needed. Dead squares, frozen 2x2 blocks and freeze-locked boxes prune
// states no push can save. If the optimal search exhausts its work budget the room is retried with a weighted
// estimate, then a greedy one, so the entry still answers; every budget is a count of expansions, never the
// clock, so the two graded runs answer byte for byte alike.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Directions: [dx, dy, walk letter, push letter]. */
const DIRS: [number, number, string, string][] = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

/** A level in XSB text; squares outside the rows and on the outermost frame count as walls. */
function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x]!;
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

/**
 * Push distances from every square to each goal for a box alone (other boxes ignored): dist[g][s] is the fewest
 * pushes that bring a box from s to goals[g], or -1 when no push does. Reverse BFS over pulls from the goal.
 */
function pushDistances(lv: Level): Int32Array[] {
  const { w, wall } = lv;
  const size = wall.length;
  return lv.goals.map((g) => {
    const dist = new Int32Array(size).fill(-1);
    dist[g] = 0;
    const queue = [g];
    for (let qi = 0; qi < queue.length; qi++) {
      const t = queue[qi]!;
      for (const [dx, dy] of DIRS) {
        const d = dy * w + dx;
        const s = t - d;
        if (s < 0 || s >= size || dist[s] !== -1 || wall[s] || wall[s - d]) continue;
        dist[s] = dist[t]! + 1;
        queue.push(s);
      }
    }
    return dist;
  });
}

interface Reach {
  seen: Int32Array;
  stack: Int32Array;
  stamp: number;
}

/** Marks the squares the player reaches from `from` around walls and boxes; returns the region's lowest square. */
function reach(lv: Level, occupied: Uint8Array, from: number, rc: Reach): number {
  const { w, wall } = lv;
  const stamp = ++rc.stamp;
  let low = from;
  let n = 0;
  rc.stack[n++] = from;
  rc.seen[from] = stamp;
  while (n > 0) {
    const p = rc.stack[--n]!;
    if (p < low) low = p;
    const q0 = p - w;
    const q1 = p + w;
    const q2 = p - 1;
    const q3 = p + 1;
    if (!wall[q0] && !occupied[q0] && rc.seen[q0] !== stamp) { rc.seen[q0] = stamp; rc.stack[n++] = q0; }
    if (!wall[q1] && !occupied[q1] && rc.seen[q1] !== stamp) { rc.seen[q1] = stamp; rc.stack[n++] = q1; }
    if (!wall[q2] && !occupied[q2] && rc.seen[q2] !== stamp) { rc.seen[q2] = stamp; rc.stack[n++] = q2; }
    if (!wall[q3] && !occupied[q3] && rc.seen[q3] !== stamp) { rc.seen[q3] = stamp; rc.stack[n++] = q3; }
  }
  return low;
}

/** Whether the box just pushed to `t` sits in a frozen 2x2 block of walls and boxes with a box off its goal. */
function frozenBlock(lv: Level, occupied: Uint8Array, goal: Uint8Array, t: number): boolean {
  const { w, wall } = lv;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (!(wall[a] || occupied[a]) || !(wall[b] || occupied[b]) || !(wall[c] || occupied[c]) || !(wall[d] || occupied[d])) continue;
    if ((occupied[a] && !goal[a]) || (occupied[b] && !goal[b]) || (occupied[c] && !goal[c]) || (occupied[d] && !goal[d])) return true;
  }
  return false;
}

/**
 * Freeze deadlock: a box that cannot move along either axis (each axis blocked on a side by a wall or an
 * already frozen box) can never move again; iterate to a fixed point. True when some frozen box is off a goal.
 */
function freezeDeadlock(lv: Level, boxes: number[], occupied: Uint8Array, goal: Uint8Array): boolean {
  const { w, wall } = lv;
  const frozen = new Uint8Array(wall.length);
  let changed = true;
  while (changed) {
    changed = false;
    for (const b of boxes) {
      if (frozen[b]) continue;
      const bx = b + 1;
      const bnx = b - 1;
      const by = b + w;
      const bny = b - w;
      const xBlocked = wall[bx] || frozen[bx] || wall[bnx] || frozen[bnx];
      const yBlocked = wall[by] || frozen[by] || wall[bny] || frozen[bny];
      if (xBlocked && yBlocked) {
        frozen[b] = 1;
        changed = true;
      }
    }
  }
  void occupied;
  return boxes.some((b) => frozen[b] && !goal[b]);
}

/** Min-cost matching of boxes to goals over push distances (subset DP); -1 when some box reaches no goal. */
function matching(boxes: number[], dists: Int32Array[]): number {
  const n = boxes.length;
  const m = dists.length;
  if (n === 0) return 0;
  const full = (1 << m) - 1;
  const dp = new Float64Array(1 << m).fill(Infinity);
  dp[0] = 0;
  for (let mask = 0; mask <= full; mask++) {
    const cur = dp[mask]!;
    if (cur === Infinity) continue;
    let k = 0;
    for (let bits = mask; bits; bits &= bits - 1) k++;
    if (k >= n) continue;
    const b = boxes[k]!;
    for (let j = 0; j < m; j++) {
      if (mask & (1 << j)) continue;
      const d = dists[j]![b]!;
      if (d < 0) continue;
      const nm = mask | (1 << j);
      if (cur + d < dp[nm]!) dp[nm] = cur + d;
    }
  }
  const v = dp[full]!;
  return v === Infinity ? -1 : v;
}

/** The shortest walk from `from` to `to` around walls and boxes, in lower LURD letters. */
function walkTo(lv: Level, occupied: Uint8Array, from: number, to: number): string {
  const { w, wall } = lv;
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array<string>(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const q = p + dy * w + dx;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = letter;
      queue.push(q);
    }
  }
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
  return s;
}

interface Solved {
  pushes: number;
  moves: string;
}

/** Work spent so far across the whole run, in expansions; the budgets below are counts, never the clock. */
let workSpent = 0;
const WORK_TOTAL = 10_000_000;

/**
 * A* over pushes for one (sub-)level. weight 1 is push-optimal; a larger weight trades pushes for speed when a
 * room resists. At most `cap` expansions. Null when the search gives up.
 */
function solvePushes(lv: Level, weight: number, cap: number): Solved | null {
  const { w, wall, goals } = lv;
  const size = wall.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const dists = pushDistances(lv);
  const live = new Uint8Array(size);
  for (let s = 0; s < size; s++) if (dists.some((d) => d[s]! >= 0)) live[s] = 1;
  if (lv.boxes.some((b) => !live[b])) return null;
  const h0 = matching(lv.boxes, dists);
  if (h0 < 0) return null;

  const occupied = new Uint8Array(size);
  // Two stamp sets: rcA holds the region of the state being expanded, rcB the region of each child it makes.
  const rcA: Reach = { seen: new Int32Array(size), stack: new Int32Array(size), stamp: 0 };
  const rcB: Reach = { seen: new Int32Array(size), stack: new Int32Array(size), stamp: 0 };
  const keyOf = (boxes: number[], low: number): string => String.fromCharCode(low, ...boxes);
  const solved = (boxes: number[]): boolean => boxes.every((b) => goal[b]);

  const sBoxes: number[][] = [lv.boxes];
  const sPlayer: number[] = [lv.player];
  const sG: number[] = [0];
  const sParent: number[] = [-1];
  for (const b of lv.boxes) occupied[b] = 1;
  const low0 = reach(lv, occupied, lv.player, rcA);
  for (const b of lv.boxes) occupied[b] = 0;
  const closed = new Map<string, number>([[keyOf(lv.boxes, low0), 0]]);

  // Binary heap on (f, then deeper g, then insertion order): deterministic on every run.
  const heapF: number[] = [h0 * weight];
  const heapS: number[] = [0];
  let seq = 0;
  const heapSeq: number[] = [seq++];
  const less = (i: number, j: number): boolean => {
    if (heapF[i] !== heapF[j]) return heapF[i]! < heapF[j]!;
    const gi = sG[heapS[i]!]!;
    const gj = sG[heapS[j]!]!;
    if (gi !== gj) return gi > gj;
    return heapSeq[i]! < heapSeq[j]!;
  };
  const swap = (i: number, j: number): void => {
    [heapF[i], heapF[j]] = [heapF[j]!, heapF[i]!];
    [heapS[i], heapS[j]] = [heapS[j]!, heapS[i]!];
    [heapSeq[i], heapSeq[j]] = [heapSeq[j]!, heapSeq[i]!];
  };
  const pushHeap = (f: number, s: number): void => {
    let i = heapF.length;
    heapF.push(f);
    heapS.push(s);
    heapSeq.push(seq++);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(i, p)) break;
      swap(i, p);
      i = p;
    }
  };
  const popHeap = (): number => {
    const top = heapS[0]!;
    const lf = heapF.pop()!;
    const ls = heapS.pop()!;
    const lq = heapSeq.pop()!;
    if (heapF.length > 0) {
      heapF[0] = lf;
      heapS[0] = ls;
      heapSeq[0] = lq;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heapF.length) break;
        if (c + 1 < heapF.length && less(c + 1, c)) c++;
        if (!less(c, i)) break;
        swap(i, c);
        i = c;
      }
    }
    return top;
  };

  let found = solved(lv.boxes) ? 0 : -1;
  let expansions = 0;
  while (found < 0 && heapF.length > 0) {
    if (expansions >= cap || workSpent >= WORK_TOTAL) return null;
    const s = popHeap();
    const boxes = sBoxes[s]!;
    const g = sG[s]!;
    for (const b of boxes) occupied[b] = 1;
    reach(lv, occupied, sPlayer[s]!, rcA);
    const here = rcA.stamp;
    if (solved(boxes)) {
      found = s;
      for (const b of boxes) occupied[b] = 0;
      break;
    }
    expansions++;
    workSpent++;
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi]!;
      for (const [dx, dy] of DIRS) {
        const d = dy * w + dx;
        const t = b + d;
        if (rcA.seen[b - d] !== here || wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const dead = frozenBlock(lv, occupied, goal, t);
        let next: number[] | null = null;
        let key = '';
        let h = -1;
        if (!dead) {
          next = boxes.slice();
          next[bi] = t;
          next.sort((a, b2) => a - b2);
          if (!freezeDeadlock(lv, next, occupied, goal)) {
            const low = reach(lv, occupied, b, rcB);
            key = keyOf(next, low);
            const seenG = closed.get(key);
            if (seenG === undefined || seenG > g + 1) h = matching(next, dists);
          }
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (next === null || h < 0 || key === '') continue;
        closed.set(key, g + 1);
        const t2 = sBoxes.length;
        sBoxes.push(next);
        sPlayer.push(b);
        sG.push(g + 1);
        sParent.push(s);
        pushHeap(g + 1 + weight * h, t2);
      }
    }
    for (const b of boxes) occupied[b] = 0;
  }
  if (found < 0) return null;
  // Reconstruct: walk to each push's stand square, then push.
  const path: number[] = [];
  for (let s = found; s > 0; s = sParent[s]!) path.push(s);
  path.reverse();
  let moves = '';
  let player = lv.player;
  let boxes = lv.boxes;
  for (const s of path) {
    const nextBoxes = sBoxes[s]!;
    const from = boxes.find((b) => !nextBoxes.includes(b))!;
    const to = nextBoxes.find((b) => !boxes.includes(b))!;
    const d = to - from;
    const dir = DIRS.find(([dx, dy]) => dy * w + dx === d)!;
    for (const b of boxes) occupied[b] = 1;
    moves += walkTo(lv, occupied, player, from - d);
    for (const b of boxes) occupied[b] = 0;
    moves += dir[3];
    player = from;
    boxes = nextBoxes;
  }
  return { pushes: path.length, moves };
}

/** One room, retried with a greedier estimate when the optimal search runs out of work. */
function solveRoom(lv: Level): Solved | null {
  return (
    solvePushes(lv, 1, 700_000) ??
    solvePushes(lv, 3, 700_000) ??
    solvePushes(lv, 10, 700_000)
  );
}

interface Room {
  squares: number[];
  boxes: number[];
  goals: number[];
  done?: boolean;
}

/**
 * The level's rooms: cut the dead squares of the player's row (the hallway) out of the floor; each part left
 * that holds a box is a room. Null when the level is no set of rooms.
 */
function roomsOf(lv: Level): Room[] | null {
  const { w, wall, goals, boxes, player } = lv;
  const dists = pushDistances(lv);
  const live = new Uint8Array(wall.length);
  for (let s = 0; s < wall.length; s++) if (dists.some((d) => d[s]! >= 0)) live[s] = 1;
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && !live[i]) cut[i] = 1;
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const [dx, dy] of DIRS) {
        const q = squares[i]! + dy * w + dx;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [] });
  }
  for (const b of boxes) rooms[part[b]!]!.boxes.push(b);
  for (const g of goals) if (part[g] >= 0) rooms[part[g]!]!.goals.push(g);
  if (goals.some((g) => part[g] < 0) || rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/** The square the player stands on after `moves` from `from` on a board `w` wide. */
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves.toLowerCase()) p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  return p;
}

/** Solve a level room by room; fall back to the whole level when it is no set of rooms. */
function solveLevel(text: string): string {
  const lv = parseLevel(text);
  const rooms = roomsOf(lv);
  if (rooms === null) return solveRoom(lv)?.moves ?? '';
  let player = lv.player;
  let moves = '';
  for (const room of rooms) {
    const wall = lv.wall.slice();
    for (const other of rooms)
      if (other !== room) for (const b of other.done ? other.goals : other.boxes) wall[b] = 1;
    const sub: Level = {
      w: lv.w,
      h: lv.h,
      wall,
      goals: room.goals.slice().sort((a, b) => a - b),
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player,
    };
    const solved = solveRoom(sub);
    if (solved === null) return '';
    moves += solved.moves;
    player = endOf(lv.w, player, solved.moves);
    room.done = true;
  }
  return moves;
}

for (const id of readdirSync(I).sort()) {
  const moves = solveLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
