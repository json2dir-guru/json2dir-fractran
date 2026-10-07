#!/usr/bin/env python3
"""fracc - compile a small register-machine language to a FRACTRAN program.

This is the standard Minsky-machine -> FRACTRAN construction: every machine state
is a prime, every register is a prime whose exponent is the register's value, and
a state with a guarded transfer becomes two fractions

    next * (added registers) / (state * (tested registers)),   else / state

so the first applicable fraction is always one of the current state's.  Bulk loops
(move, clear, mul, div, ...) are two-state cycles; an interpreter with cycle
detection can fast-forward them.

Usage: fracc.py SOURCE.rm > PROGRAM.fractran
       fracc.py SOURCE.rm --run < input       (reference simulation, for testing)

Source language (one statement per line, '#' starts a comment):

  fixed R p            register R is the prime p (used for IN=2 and OUT=3)
  reg R1 R2 ...        declare registers
  macro NAME a b ...   textual macro; $a in the body is replaced by the argument,
  ...                  @x is a label local to one expansion
  endmacro
  sub NAME ... end     subroutine: 'call NAME' / 'ret' (return dispatch on a register)
  main ... end         entry point

Statements:
  inc R [n]  dec R [n]  clear R  move A B[:k] ...  copy A B[:k] ...  mul A n
  div A n Q            Q += A div n, A = A mod n
  bits A hi lo D[:k].. for each bit position i=hi..lo: if A >= 2^i, A -= 2^i and
                       D += 2^(i-lo)*k   (moves a small value bit by bit; no D = zap)
  sub2 A B             while A > 0 and B > 0: decrement both
  emit "text"          OUT = OUT * 1000^len + bytes (one base-1000 digit per byte)
  call S  ret  goto L  L:  halt  fail "message"
  if R OP n ... [else ...] end        OP is one of >= > <= < == !=
  while R OP n ... end
  switch R [consume]   case v w lo..hi ...  default  end   (no fall-through;
                       with 'consume', a case made only of single values leaves R = 0)
"""
import sys

INF = float('inf')
import os
TRACE = os.environ.get('FRACC_TRACE')


class Err(Exception):
    pass


# ---------------------------------------------------------------- lexer

ESC = {'n': 10, 't': 9, 'r': 13, '0': 0, '\\': 92, '"': 34, "'": 39}


def read_escape(s, i):
    c = s[i]
    if c == 'x':
        return int(s[i + 1:i + 3], 16), i + 3
    if c not in ESC:
        raise Err('bad escape \\' + c)
    return ESC[c], i + 1


def tokenize(line):
    toks = []
    i = 0
    n = len(line)
    while i < n:
        c = line[i]
        if c in ' \t\r':
            i += 1
        elif c == '#':
            break
        elif c == '"':
            i += 1
            out = bytearray()
            while True:
                if i >= n:
                    raise Err('unterminated string')
                c = line[i]
                if c == '"':
                    i += 1
                    break
                if c == '\\':
                    v, i = read_escape(line, i + 1)
                    out.append(v)
                else:
                    out += c.encode('utf-8')
                    i += 1
            toks.append(('str', bytes(out)))
        elif c == "'":
            if line[i + 1] == '\\':
                v, j = read_escape(line, i + 2)
            else:
                v, j = ord(line[i + 1]), i + 2
            if j >= n or line[j] != "'":
                raise Err('bad character literal')
            toks.append(('int', v))
            i = j + 1
        elif line.startswith('..', i):
            toks.append(('op', '..'))
            i += 2
        else:
            j = i
            while j < n and line[j] not in ' \t\r"\'#' and not line.startswith('..', j):
                j += 1
            w = line[i:j]
            i = j
            try:
                toks.append(('int', int(w, 0)))
            except ValueError:
                toks.append(('word', w))
    return toks


# ---------------------------------------------------------------- parser

BLOCK_OPEN = {'if', 'while', 'switch'}


class Parser:
    def __init__(self, text):
        self.macros = {}
        self.subs = {}
        self.sub_order = []
        self.main = None
        self.fixed = {}
        self.regs = set()
        self.nexp = 0
        lines = []
        for no, raw in enumerate(text.split('\n'), 1):
            try:
                t = tokenize(raw)
            except Err as e:
                raise Err('line %d: %s' % (no, e))
            if t:
                lines.append((no, t))
        self.lines = lines
        self.pos = 0
        self.queue = []          # expanded macro lines waiting to be parsed
        self.top()

    def next_line(self):
        if self.queue:
            return self.queue.pop(0)
        if self.pos >= len(self.lines):
            return None
        l = self.lines[self.pos]
        self.pos += 1
        return l

    def top(self):
        while True:
            l = self.next_line()
            if l is None:
                break
            no, t = l
            w = t[0][1]
            if w == 'fixed':
                self.fixed[t[1][1]] = t[2][1]
                self.regs.add(t[1][1])
            elif w == 'reg':
                for x in t[1:]:
                    self.regs.add(x[1])
            elif w == 'macro':
                name = t[1][1]
                params = [x[1] for x in t[2:]]
                body = []
                while True:
                    l2 = self.next_line()
                    if l2 is None:
                        raise Err('line %d: unterminated macro' % no)
                    if l2[1][0] == ('word', 'endmacro'):
                        break
                    body.append(l2)
                self.macros[name] = (params, body)
            elif w == 'sub':
                name = t[1][1]
                body, end = self.block(('end',))
                self.subs[name] = body
                self.sub_order.append(name)
            elif w == 'main':
                self.main, end = self.block(('end',))
            else:
                raise Err('line %d: unexpected %r at top level' % (no, w))

    def expand(self, no, t):
        params, body = self.macros[t[0][1]]
        args = t[1:]
        if len(args) != len(params):
            raise Err('line %d: macro %s takes %d arguments' % (no, t[0][1], len(params)))
        self.nexp += 1
        out = []
        for bno, bt in body:
            nt = []
            for tok in bt:
                if tok[0] == 'word':
                    w = tok[1]
                    whole = None
                    for p, a in zip(params, args):
                        if w == '$' + p:
                            whole = a
                    if whole is not None:
                        nt.append(whole)
                        continue
                    for p, a in sorted(zip(params, args), key=lambda x: -len(x[0])):
                        if '$' + p in w:
                            w = w.replace('$' + p, str(a[1]))
                    if '@' in w:
                        w = w.replace('@', '@%d_' % self.nexp)
                    nt.append(('word', w))
                else:
                    nt.append(tok)
            out.append((bno, nt))
        self.queue[0:0] = out

    def block(self, terminators):
        """Parse statements until a line whose first word is in terminators."""
        stmts = []
        while True:
            l = self.next_line()
            if l is None:
                raise Err('unexpected end of file (missing %s)' % '/'.join(terminators))
            no, t = l
            head = t[0][1] if t[0][0] == 'word' else None
            if head in terminators:
                return stmts, (no, t)
            if head in self.macros:
                self.expand(no, t)
                continue
            if head and head.endswith(':') and len(t) == 1:
                stmts.append(('label', head[:-1], no))
                continue
            if head == 'if':
                cond = self.cond(no, t[1:])
                then, end = self.block(('else', 'end'))
                els = []
                if end[1][0][1] == 'else':
                    els, end = self.block(('end',))
                stmts.append(('switch', cond[0], False, [(cond[1], then)], els, no))
            elif head == 'while':
                cond = self.cond(no, t[1:])
                body, end = self.block(('end',))
                stmts.append(('while', cond, body, no))
            elif head == 'switch':
                reg = t[1][1]
                consume = len(t) > 2 and t[2][1] == 'consume'
                cases = []
                default = []
                cur = None
                l2 = self.next_line()
                while True:
                    if l2 is None:
                        raise Err('line %d: unterminated switch' % no)
                    no2, t2 = l2
                    h2 = t2[0][1]
                    if h2 == 'case':
                        vals = self.values(no2, t2[1:])
                        body, end = self.block(('case', 'default', 'end'))
                        cases.append((vals, body))
                        l2 = end
                    elif h2 == 'default':
                        default, end = self.block(('case', 'default', 'end'))
                        l2 = end
                    elif h2 == 'end':
                        break
                    else:
                        raise Err('line %d: expected case/default' % no2)
                stmts.append(('switch', reg, consume, cases, default, no))
            else:
                stmts.append(('op', head, t[1:], no))

    def values(self, no, toks):
        vals = []
        i = 0
        while i < len(toks):
            if toks[i][0] != 'int':
                raise Err('line %d: bad case value %r' % (no, toks[i][1]))
            lo = toks[i][1]
            if i + 2 < len(toks) + 1 and i + 1 < len(toks) and toks[i + 1] == ('op', '..'):
                hi = toks[i + 2][1]
                vals.append((lo, hi))
                i += 3
            else:
                vals.append((lo, lo))
                i += 1
        return vals

    def cond(self, no, toks):
        if len(toks) != 3 or toks[2][0] != 'int':
            raise Err('line %d: condition must be: REG OP number' % no)
        r, op, n = toks[0][1], toks[1][1], toks[2][1]
        table = {
            '>=': [(n, INF)], '>': [(n + 1, INF)],
            '<': [(0, n - 1)] if n > 0 else [], '<=': [(0, n)],
            '==': [(n, n)], '!=': ([(0, n - 1)] if n > 0 else []) + [(n + 1, INF)],
        }
        if op not in table:
            raise Err('line %d: bad operator %s' % (no, op))
        return r, [v for v in table[op] if v[0] <= v[1]]


# ---------------------------------------------------------------- code generation

class Node:
    __slots__ = ('guard', 'add', 'then', 'els', 'keep', 'origin')

    def __init__(self, guard, add, then, els, keep, origin=None):
        self.guard, self.add, self.then, self.els, self.keep = guard, add, then, els, keep
        self.origin = origin


class Gen:
    def __init__(self, p):
        self.p = p
        self.nodes = []
        self.labels = {}
        self.ntmp = 0
        self.nlab = 0
        self.temps = []
        self.messages = []
        self.callsites = {s: [] for s in p.subs}
        self.cur_sub = None
        self.where = None

    def reg(self, r, no):
        if r not in self.p.regs and r not in self.temps:
            raise Err('line %s: unknown register %s' % (no, r))
        return r

    def tmp(self):
        self.ntmp += 1
        r = '_t%d' % self.ntmp
        self.temps.append(r)
        return r

    def label(self, hint='L'):
        self.nlab += 1
        return '.%s%d' % (hint, self.nlab)

    def here(self, lab):
        if lab in self.labels:
            raise Err('duplicate label ' + lab)
        self.labels[lab] = len(self.nodes)

    def node(self, guard=None, add=None, then='+', els=None, keep=False):
        guard = {k: v for k, v in (guard or {}).items() if v}
        add = {k: v for k, v in (add or {}).items() if v}
        if set(guard) & set(add):
            raise Err('register both tested and added in one step')
        if els is None and guard:
            els = '.panic'
        self.nodes.append(Node(guard, add, then, els, keep, self.where))

    def goto(self, lab):
        self.node(then=lab)

    # -- bulk loops: two-state cycles
    def loop(self, guard, add):
        top, end = self.label('loop'), self.label('end')
        self.here(top)
        self.node(guard, add, then='+', els=end)
        self.node(then=top, keep=True)
        self.here(end)

    def dests(self, toks, no, base=1):
        out = {}
        for t in toks:
            w = t[1]
            k = 1
            if ':' in w:
                w, ks = w.split(':')
                k = int(ks, 0)
            out[self.reg(w, no)] = out.get(self.reg(w, no), 0) + k * base
        return out

    # -- multi-way branch on a register by binary search over interval boundaries
    def switch(self, r, consume, cases, default_lab):
        points = {0}
        for vals, lab in cases:
            for lo, hi in vals:
                points.add(lo)
                if hi != INF:
                    points.add(hi + 1)
        pts = sorted(points)
        segs = []
        for i, a in enumerate(pts):
            b = pts[i + 1] - 1 if i + 1 < len(pts) else INF
            target, single = default_lab, False
            for vals, lab in cases:
                if any(lo <= a and b <= hi for lo, hi in vals):
                    target = lab
                    single = consume and all(lo == hi for lo, hi in vals)
                    break
            if segs and segs[-1][2] == target and not single and not segs[-1][3]:
                segs[-1][1] = b
            else:
                segs.append([a, b, target, single])
        x = self.tmp()

        def tree(ss, off):
            if len(ss) == 1:
                a, b, target, single = ss[0]
                if single:
                    self.node({r: a - off, x: off}, {}, then=target)
                elif off:
                    self.node({x: off}, {r: off}, then=target)
                else:
                    self.goto(target)
                return
            mid = len(ss) // 2
            m = ss[mid][0]
            lo_lab = self.label('lo')
            hi_lab = self.label('hi')
            self.node({r: m - off}, {x: m - off}, then=hi_lab, els=lo_lab)
            self.here(lo_lab)
            tree(ss[:mid], off)
            self.here(hi_lab)
            tree(ss[mid:], m)
        tree(segs, 0)

    def stmts(self, ss):
        for s in ss:
            self.stmt(s)

    def stmt(self, s):
        kind = s[0]
        self.where = (self.cur_sub, s[-1])
        if kind == 'label':
            self.here(s[1])
        elif kind == 'switch':
            _, r, consume, cases, default, no = s
            self.reg(r, no)
            end = self.label('send')
            dlab = self.label('default')
            labs = []
            for vals, body in cases:
                labs.append((vals, self.label('case')))
            self.switch(r, consume, labs, dlab)
            for (vals, lab), (_, body) in zip(labs, cases):
                self.here(lab)
                self.stmts(body)
                self.goto(end)
            self.here(dlab)
            self.stmts(default)
            self.here(end)
        elif kind == 'while':
            _, (r, vals), body, no = s
            self.reg(r, no)
            top, bl, end = self.label('wtop'), self.label('wbody'), self.label('wend')
            self.here(top)
            self.switch(r, False, [(vals, bl)], end)
            self.here(bl)
            self.stmts(body)
            self.goto(top)
            self.here(end)
        else:
            self.op(s)

    def op(self, s):
        _, name, a, no = s
        R = lambda i: self.reg(a[i][1], no)
        N = lambda i, d=1: a[i][1] if len(a) > i else d
        if name == 'inc':
            self.node({}, {R(0): N(1)})
        elif name == 'dec':
            self.node({R(0): N(1)}, {})
        elif name == 'clear':
            for i in range(len(a)):
                self.loop({R(i): 1}, {})
        elif name == 'move':
            self.loop({R(0): 1}, self.dests(a[1:], no))
        elif name == 'copy':
            t = self.tmp()
            d = self.dests(a[1:], no)
            d[t] = 1
            self.loop({R(0): 1}, d)
            self.loop({t: 1}, {R(0): 1})
        elif name == 'mul':
            t = self.tmp()
            self.loop({R(0): 1}, {t: 1})
            self.loop({t: 1}, {R(0): N(1)})
        elif name == 'div':
            self.loop({R(0): N(1)}, {R(2): 1})
        elif name == 'sub2':
            self.loop({R(0): 1, R(1): 1}, {})
        elif name == 'bits':
            r, hi, lo = R(0), a[1][1], a[2][1]
            for i in range(hi, lo - 1, -1):
                self.node({r: 1 << i}, self.dests(a[3:], no, 1 << (i - lo)), then='+', els='+')
        elif name == 'emit':
            data = a[0][1]
            if not data:
                return
            v = 0
            for b in data:
                v = v * 1000 + b
            t = self.tmp()
            self.loop({'OUT': 1}, {t: 1})
            self.loop({t: 1}, {'OUT': 1000 ** len(data)})
            self.node({}, {'OUT': v})
        elif name == 'call':
            sub = a[0][1]
            if sub not in self.p.subs:
                raise Err('line %d: unknown sub %s' % (no, sub))
            sites = self.callsites[sub]
            ret = self.label('ret')
            sites.append(ret)
            self.node({}, {'_ret_' + sub: len(sites)}, then='.sub_' + sub)
            self.here(ret)
        elif name == 'ret':
            if not self.cur_sub:
                raise Err('line %d: ret outside sub' % no)
            self.goto('.retd_' + self.cur_sub)
        elif name == 'goto':
            self.goto(a[0][1])
        elif name == 'halt':
            self.goto('.halt')
        elif name == 'fail':
            msg = a[0][1]
            if msg not in self.messages:
                self.messages.append(msg)
            self.node({}, {'_err': self.messages.index(msg) + 1}, then='.fail')
        else:
            raise Err('line %d: unknown statement %s' % (no, name))

    def ret_dispatch(self, sub, sites):
        r = '_ret_' + sub
        self.temps.append(r)

        def tree(vals, off):
            if len(vals) == 1:
                v = vals[0]
                self.node({r: v - off}, {}, then=sites[v - 1])
                return
            mid = len(vals) // 2
            m = vals[mid]
            lo_lab = self.label('rlo')
            self.node({r: m - off}, {}, then='+', els=lo_lab)
            tree(vals[mid:], m)
            self.here(lo_lab)
            tree(vals[:mid], off)
        if not sites:
            return
        tree(list(range(1, len(sites) + 1)), 0)

    def chunk(self):
        self.nodes, self.labels = [], {}

    def build(self):
        p = self.p
        self.temps.append('_err')
        chunks = {}
        # subroutine bodies first (their dispatchers come later, once every call
        # site is known), then main and the fail handler, then the dispatchers.
        for s in p.sub_order:
            self.chunk()
            self.cur_sub = s
            self.here('.sub_' + s)
            self.stmts(p.subs[s])
            self.goto('.retd_' + s)
            chunks['body', s] = (self.nodes, self.labels)
        self.cur_sub = None
        self.chunk()
        self.here('.main')
        self.stmts(p.main)
        self.goto('.halt')
        # fail handler: clear the output, write the message, empty IN, halt
        self.here('.panic')
        self.node({}, {'_err': len(self.messages) + 1}, then='.fail')
        self.here('.fail')
        self.loop({'OUT': 1}, {})
        self.node({}, {'OUT': 1})
        cases = []
        msgs = self.messages + [b'internal error']
        for i, m in enumerate(msgs):
            cases.append(([(i + 1, i + 1)], [('op', 'emit', [('str', b'json2dir: ' + m + b'\n')], 0),
                                              ('op', 'goto', [('word', '.fin')], 0)]))
        self.stmt(('switch', '_err', False, cases, [], 0))
        self.here('.fin')
        self.loop({'IN': 1}, {})
        self.goto('.halt')
        chunks['main'] = (self.nodes, self.labels)
        for s in p.sub_order:
            self.chunk()
            self.here('.retd_' + s)
            self.ret_dispatch(s, self.callsites[s])
            chunks['disp', s] = (self.nodes, self.labels)
        order = []
        for s in p.sub_order:
            order += [chunks['body', s], chunks['disp', s]]
        order.append(chunks['main'])
        nodes, labels = [], {}
        for cn, cl in order:
            for k, v in cl.items():
                if k in labels:
                    raise Err('duplicate label ' + k)
                labels[k] = v + len(nodes)
            nodes += cn
        self.nodes, self.labels = nodes, labels

    def link(self):
        nodes, labels = self.nodes, self.labels
        n = len(nodes)
        HALT = n
        labels['.halt'] = HALT

        def res(t, i):
            if t == '+':
                return i + 1
            if t not in labels:
                raise Err('undefined label ' + t)
            return labels[t]
        for i, nd in enumerate(nodes):
            nd.then = res(nd.then, i)
            if nd.els is not None:
                nd.els = res(nd.els, i)

        def pure(i):
            return i < n and not nodes[i].guard and not nodes[i].add and not nodes[i].keep

        def follow(t):
            seen = set()
            while pure(t) and t not in seen:
                seen.add(t)
                t = nodes[t].then
            return t
        for i, nd in enumerate(nodes):
            t = follow(nd.then)
            if t != i:
                nd.then = t
            if nd.els is not None:
                t = follow(nd.els)
                if t != i:
                    nd.els = t
        entry = follow(labels['.main'])
        # keep reachable nodes, in their original order
        reach = set()
        stack = [entry]
        while stack:
            i = stack.pop()
            if i in reach or i == HALT:
                continue
            reach.add(i)
            stack.append(nodes[i].then)
            if nodes[i].els is not None:
                stack.append(nodes[i].els)
        order = [i for i in range(n) if i in reach]
        for i in order:
            nd = nodes[i]
            if nd.then == i or nd.els == i:
                raise Err('self loop at node %d' % i)
        self.order, self.entry, self.HALT = order, entry, HALT

    def assign(self):
        gen = primes()
        used = {}
        for r, pr in self.p.fixed.items():
            used[r] = pr
        taken = set(used.values())

        def nextp():
            while True:
                q = next(gen)
                if q not in taken:
                    return q
        sp = {}
        for i in self.order:
            sp[i] = nextp()
        sp[self.HALT] = nextp()
        for r in sorted(self.p.regs | set(self.temps)):
            if r not in used:
                used[r] = nextp()
        self.sp, self.rp = sp, used

    def fractions(self):
        out = []
        sp, rp = self.sp, self.rp
        for i in self.order:
            nd = self.nodes[i]
            num = {sp[nd.then]: 1}
            for r, k in nd.add.items():
                num[rp[r]] = num.get(rp[r], 0) + k
            den = {sp[i]: 1}
            for r, k in nd.guard.items():
                den[rp[r]] = k
            out.append((num, den, i))
            if nd.guard:
                out.append(({sp[nd.els]: 1}, {sp[i]: 1}, i))
        # start fraction: the initial number is 2^input; take one 2 and enter main
        # (main begins with 'inc IN').
        out.append(({sp[self.entry]: 1}, {rp['IN']: 1}, None))
        self.fracs = out
        return out


def primes():
    found = []
    q = 2
    while True:
        if all(q % f for f in found if f * f <= q):
            found.append(q)
            yield q
        q += 1


def fmt(m):
    if not m:
        return '1'
    return '*'.join(str(p) if e == 1 else '%d^%d' % (p, e) for p, e in sorted(m.items()))


def compile_source(text):
    p = Parser(text)
    g = Gen(p)
    g.build()
    g.link()
    g.assign()
    g.fractions()
    return g


def write_program(g, f, srcname):
    f.write('# json2dir in FRACTRAN, generated by tools/fracc.py from %s -- do not edit.\n' % srcname)
    f.write('# Input: the initial number is 2^E with E = 256^n + sum(byte_i * 256^i).\n')
    f.write('# Output: the exponent of 3 at the end is 1 followed by one base-1000 digit per byte.\n')
    f.write('# %d fractions, %d state primes, %d registers.\n' % (len(g.fracs), len(g.order) + 1, len(g.rp)))
    for num, den, _ in g.fracs:
        f.write('%s/%s\n' % (fmt(num), fmt(den)))


# ---------------------------------------------------------------- reference simulator

def simulate(g, data, limit=10 ** 9):
    """Run the fraction list with exponent vectors; fast-forwards two-state loops."""
    E = 256 ** len(data)
    for i, b in enumerate(data):
        E += b << (8 * i)
    regs = {2: E}
    by_state = {}
    start = None
    for num, den, i in g.fracs:
        if i is None:
            start = (num, den)
            continue
        s = g.sp[i]
        by_state.setdefault(s, []).append((num, den))
    state_set = set(by_state) | {g.sp[g.HALT]}
    inv0 = {v: k for k, v in g.sp.items()}
    state = None
    steps = 0
    ffsteps = 0

    def applicable(den):
        return all(regs.get(p, 0) >= e for p, e in den.items())

    def apply(num, den, k=1):
        for p, e in den.items():
            v = regs.get(p, 0) - e * k
            if v:
                regs[p] = v
            else:
                regs.pop(p, None)
        for p, e in num.items():
            regs[p] = regs.get(p, 0) + e * k
    # start
    apply(*start)
    state = [p for p in regs if p in state_set][0]
    while steps < limit:
        fl = by_state.get(state)
        if fl is None:
            break
        for num, den in fl:
            if applicable(den):
                break
        else:
            raise Err('no fraction applies in state %d' % state)
        nxt = [p for p in num if p in state_set][0]
        if TRACE:
            nd = g.nodes[inv0[state]]
            sys.stderr.write('%s %s B=%s\n' % (inv0[state], nd.origin, regs.get(g.rp['B'], 0)))
        apply(num, den)
        steps += 1
        # two-state loop fast-forward
        back = by_state.get(nxt)
        if back and len(back) == 1 and back[0][0] == {state: 1} and back[0][1] == {nxt: 1} and fl[0] is not None and (num, den) == fl[0]:
            k = min([(regs.get(p, 0) // e) for p, e in den.items() if p != state] or [0])
            if k > 0:
                # apply (bounce, loop) k more times
                d2 = {p: e for p, e in den.items() if p != state}
                n2 = {p: e for p, e in num.items() if p != nxt}
                apply(n2, d2, k)
                ffsteps += 2 * k
        state = nxt
    inv = {v: k for k, v in g.sp.items()}
    if state != g.sp[g.HALT]:
        nd = g.nodes[inv[state]]
        sys.stderr.write('stopped in node %d from %s\n' % (inv[state], nd.origin))
    if 2 in regs and regs[2]:
        sys.stderr.write('warning: halted with IN = %d in state %s\n' % (regs[2], state))
    return regs, steps


def decode_out(o):
    s = str(o)
    if not s.startswith('1') or (len(s) - 1) % 3:
        raise Err('bad output register')
    return bytes(int(s[i:i + 3]) for i in range(1, len(s), 3))


def main():
    args = sys.argv[1:]
    if not args:
        sys.stderr.write(__doc__)
        sys.exit(2)
    src = args[0]
    with open(src, encoding='utf-8') as f:
        text = f.read()
    try:
        g = compile_source(text)
    except Err as e:
        sys.stderr.write('fracc: %s\n' % e)
        sys.exit(1)
    if '--run' in args:
        data = sys.stdin.buffer.read()
        regs, steps = simulate(g, data)
        sys.stderr.write('steps: %d\n' % steps)
        sys.stdout.buffer.write(decode_out(regs.get(3, 0)))
        return
    if '--stats' in args:
        sys.stderr.write('%d fractions, %d states, %d registers\n' % (len(g.fracs), len(g.order), len(g.rp)))
    import os
    write_program(g, sys.stdout, os.path.basename(src))


if __name__ == '__main__':
    main()
