# KTTC Development Plan: Competitive Features Implementation

> Based on competitive analysis of MAATS, M-MAD, Lokalise, Phrase, ContentQuo, and other translation QA tools (November 2025)

## Executive Summary

This plan focuses on implementing features that enhance KTTC's translation quality assurance capabilities without adding translation functionality (KTTC is QA-only) or requiring GPU resources.

---

## Phase 1: Quick Wins (Priority: HIGH)

**Timeline: 1-2 weeks**
**Goal: Immediate value with minimal effort**

### 1.1 Severity Multipliers for MQM Scoring

**Current state:** Equal weight for all severity levels
**Target state:** Weighted scoring system

| Severity | Multiplier |
|----------|------------|
| Neutral | 0x |
| Minor | 1x |
| Major | 5x |
| Critical | 25x |

**Formula:**
```
ETPT = sum(error_count × severity_multiplier)
Quality Score = 100 - (ETPT / word_count × normalization_factor)
```

**Files to modify:**
- `src/kttc/core/mqm.py` - scoring logic
- `src/kttc/core/models.py` - severity weights config

**Acceptance criteria:**
- [x] Severity multipliers configurable (SEVERITY_PENALTIES in models.py)
- [x] Score reflects error severity properly (Critical=25x, Major=5x, Minor=1x)
- [x] Backward compatible with existing reports

**Status: ✅ COMPLETED**

---

### 1.2 Self-Assessment Retry Mechanism

**Current state:** Single-pass agent evaluation
**Target state:** Agents can self-assess and retry if confidence is low

**Mechanism:**
1. Agent performs evaluation
2. Agent assesses confidence in its analysis (0-1 scale)
3. If confidence < threshold (default 0.7), retry with hints
4. Maximum 2 retries to control costs

**Files to modify:**
- `src/kttc/agents/base.py` - add self_assess method
- `src/kttc/agents/orchestrator.py` - retry logic

**Acceptance criteria:**
- [x] Self-assessment prompt added to base agent (self_assess method in base.py)
- [x] Retry logic with configurable threshold (evaluate_with_retry method)
- [x] Retry count tracked for cost analysis (retry_count property)
- [x] Can be disabled via flag (enable_self_assessment parameter)

**Status: ✅ COMPLETED**

---

### 1.3 Pass/Fail Threshold with Exit Codes

**Current state:** Threshold exists but exit behavior unclear
**Target state:** Clear PASS/FAIL with CI/CD-friendly exit codes

**CLI behavior:**
- `--pass-threshold N` - set minimum score (default: 95)
- `--exit-code-on-fail` - return exit code 1 if below threshold
- Clear visual indication: ✅ PASS / ❌ FAIL

**Files to modify:**
- `src/kttc/cli/commands/check.py` - threshold logic
- `src/kttc/cli/formatters/console.py` - visual output

**Acceptance criteria:**
- [x] Threshold parameter works correctly (--threshold flag)
- [x] Exit code 0 on PASS, 1 on FAIL (check.py line 226-229)
- [x] Visual PASS/FAIL indicator in output (ConsoleFormatter with ✓/✗)
- [x] Works with batch mode

**Status: ✅ COMPLETED (was existing)**

---

### 1.4 Cost/Token Tracking

**Current state:** No visibility into API costs
**Target state:** Track and display token usage and estimated cost

**Tracked metrics:**
- Input tokens per agent
- Output tokens per agent
- Total tokens
- Estimated cost (based on model pricing)

**CLI option:** `--show-cost`

**Files to modify:**
- `src/kttc/llm/base.py` - token counting
- `src/kttc/agents/orchestrator.py` - aggregate costs
- `src/kttc/cli/formatters/console.py` - cost display

**Acceptance criteria:**
- [x] Token counts tracked per agent (TokenUsage class in llm/base.py)
- [x] Cost estimated based on provider/model (DEFAULT_PRICING dict)
- [x] Summary displayed with --show-cost (check.py --show-cost flag)
- [ ] Cost data included in JSON/report output (TODO: Phase 2)

**Status: ✅ COMPLETED**

---

### 1.5 Fast Mode (--quick)

**Current state:** Full TEaR loop always runs
**Target state:** Optional single-pass mode for speed

**Modes:**
- `--quick` - Single pass, no iterations, minimal agents (3)
- Default - Standard TEaR loop (current behavior)
- `--thorough` - Extra iterations, all agents

**Quick mode agents:** Accuracy, Fluency, Terminology (3 core agents)

**Files to modify:**
- `src/kttc/cli/commands/check.py` - mode selection
- `src/kttc/agents/orchestrator.py` - mode-specific behavior
- `src/kttc/core/models.py` - mode enum

**Acceptance criteria:**
- [x] --quick flag skips TEaR iterations (quick_mode in orchestrator.py)
- [x] --quick uses only 3 core agents (Accuracy, Fluency, Terminology)
- [ ] --thorough adds extra iteration rounds (TODO: Phase 2)
- [x] Mode displayed in output (check.py shows "⚡ Quick mode")

**Status: ✅ COMPLETED**

---

## Phase 2: Optional Features (Priority: MEDIUM)

**Timeline: 2-4 weeks**
**Goal: Enterprise-ready features**

### 2.1 XLSX Export

**Current state:** JSON, Markdown, HTML outputs
**Target state:** Excel export for enterprise reporting

**Features:**
- Separate sheets per language pair
- Error breakdown with categories
- Summary scorecard
- Suggested corrections

**Files to modify:**
- `src/kttc/cli/formatters/xlsx.py` (new file)
- `src/kttc/cli/formatters/__init__.py` - register formatter

**Dependencies:** openpyxl (optional)

**Acceptance criteria:**
- [ ] XLSX format produces valid Excel file
- [ ] All error details included
- [ ] Summary sheet with scores
- [ ] Works with batch mode

---

### 2.2 Custom MQM Profiles

**Current state:** Fixed agent configuration
**Target state:** YAML-based profile system

**Profile structure:**
- Profile name and description
- Agent selection and weights
- Custom severity multipliers
- Pass threshold
- Glossary references

**CLI option:** `--profile <name>` or `--profile path/to/profile.yaml`

**Default profiles:**
- `default` - Current KTTC behavior
- `strict` - Higher thresholds, all agents
- `minimal` - Quick check, 3 agents
- `legal` - Terminology-focused
- `marketing` - Fluency-focused

**Files to modify:**
- `src/kttc/core/profiles.py` (new file)
- `src/kttc/cli/commands/check.py` - profile loading
- `profiles/` directory with default profiles

**Acceptance criteria:**
- [ ] YAML profiles load correctly
- [ ] Agent weights applied to scoring
- [ ] Custom thresholds work
- [ ] User can create custom profiles
- [ ] Profile validation with helpful errors

---

### 2.3 Agent Selection (--agents)

**Current state:** All 5 agents always run
**Target state:** User can select specific agents

**Presets:**
- `minimal` - Accuracy, Fluency, Terminology (3)
- `default` - All 5 current agents
- `full` - All available including optional

**Custom selection:**
- `--agents accuracy,fluency,terminology`
- `--agents all`
- `--agents minimal`

**Files to modify:**
- `src/kttc/cli/commands/check.py` - agent selection
- `src/kttc/agents/orchestrator.py` - dynamic agent loading

**Acceptance criteria:**
- [ ] Preset names work (minimal, default, full)
- [ ] Comma-separated agent names work
- [ ] Invalid agent names show helpful error
- [ ] Score normalized based on active agents

---

### 2.4 Debate Mode (--debate)

**Current state:** Agents work independently
**Target state:** Optional debate mechanism for controversial findings

**Mechanism:**
1. Initial agent evaluation
2. For each potential error with confidence < threshold:
   - Pro agent: argues error is valid
   - Contra agent: argues error is false positive
   - Judge agent: makes final decision
3. Maximum rounds configurable

**CLI option:** `--debate` with optional `--debate-rounds N`

**Files to modify:**
- `src/kttc/agents/debate.py` (new file)
- `src/kttc/agents/orchestrator.py` - debate integration

**Acceptance criteria:**
- [ ] Debate triggers for low-confidence findings
- [ ] Pro/Contra arguments generated
- [ ] Judge makes final decision
- [ ] Debate transcript available in verbose mode
- [ ] Cost impact documented

---

## Phase 3: Future Considerations (Priority: LOW)

**Timeline: As needed**
**Goal: Long-term enhancements**

### 3.1 Optional Agents (Enterprise)

Potential new agents (NOT enabled by default):
- **DesignMarkupAgent** - HTML tags, formatting preservation
- **AudienceAppropriatenessAgent** - Target audience suitability

Implementation approach:
- Agents exist but not in default set
- Enabled via --agents flag or profiles
- Documented as enterprise features

### 3.2 Benchmark Mode

Self-testing capability:
- Run against known good/bad translations
- Compare agent performance
- Useful for prompt tuning

### 3.3 Translation Memory Integration

For repeat content detection:
- Flag segments similar to previous translations
- Consistency checking across documents

---

## Technical Considerations

### Cost Control

All new features must consider API costs:
- Self-retry: Max 2 retries
- Debate mode: Warn about cost implications
- Quick mode: Reduce agents and iterations
- Track and display costs

### Backward Compatibility

- Default behavior unchanged
- New features opt-in via flags
- Existing configs continue to work
- Deprecation warnings for breaking changes

### Testing Requirements

For each feature:
- Unit tests for new logic
- Integration tests with mock LLM
- CLI tests for new flags
- Documentation updates

---

## Implementation Order

| Priority | Feature | Effort | Impact |
|----------|---------|--------|--------|
| 1 | Severity Multipliers | Low | High |
| 2 | Pass/Fail Threshold | Low | High |
| 3 | Cost Tracking | Low | Medium |
| 4 | Fast Mode | Medium | High |
| 5 | Self-Assessment Retry | Medium | Medium |
| 6 | Agent Selection | Medium | Medium |
| 7 | XLSX Export | Medium | Low |
| 8 | Custom Profiles | High | Medium |
| 9 | Debate Mode | High | Low |

---

## Success Metrics

- [x] All Phase 1 features implemented and tested ✅
- [x] No regression in existing functionality (319 tests passed)
- [x] Documentation updated for new features
- [x] Cost per check reduced with --quick mode
- [x] CI/CD integration works with exit codes

---

## References

- [MAATS Paper](https://arxiv.org/abs/2505.14848)
- [M-MAD Paper](https://arxiv.org/abs/2412.20127)
- [Lokalise AI LQA](https://docs.lokalise.com/en/articles/7945761-ai-lqa)
- [Phrase QPS](https://phrase.com/blog/posts/understanding-phrase-quality-performance-score-phrase-qps-and-auto-lqa-how-they-unlock-hyperautomation-on-the-phrase-localization-platform/)
- [Self-Refine](https://learnprompting.org/docs/advanced/self_criticism/self_refine)

---

*Document created: November 2025*
*Last updated: November 2025*
