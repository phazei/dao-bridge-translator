param(
    [string]$Model = "gemma-4-31b-it",
    [double]$Temperature = 0.6,
    [int]$MaxTokens = 4500
)

$sourceText = @"
第一章　『腐敗する精神』







　　　　１



　──晴れ渡るような青空が、仰向けに倒れるスバルの視界いっぱいに広がっていた。



　異世界召喚されてから、振り返って約二ヵ月半ほどが経過している。

　その間、こうした形で青空を見上げることになったのはもう何度目になるだろうか。

　入道雲が厚く日差しを遮っているが、煌々と照りつける陽光は雲の厚みを通り抜けて地上へ降り注いでいる。

　日の光に瞼の奥を焼かれながら、ふとスバルはとりとめもなく思う。

「そういえば……こっちきてから今んとこ、雨の日に遭遇したことねぇな」

　夜遅くにぱらつく小雨や、夕焼け前後の通り雨ぐらいなら何度か経験したが、一日降り続くような長雨には今のところ出くわしていない。

　ルグニカの気温は長袖で過ごすにはわずかに暑く、体感的には元の世界の六月、あるいは残暑を抜けた九月ぐらいの感覚だろうか。

　雨の少なさからして、こちらの世界の乾季というやつなのかもしれない。

「そろそろ終わりにいたしますかな？」

　寝転がって思考遊びをしているスバルに、ふいにそんな声がかけられた。

　仰向けのまま、首を持ち上げる視線の先に一人の老人が立っている。

　背の高い、黒一色の執事服を身にまとう人物だ。年齢を感じさせない鍛えられた体と、ピンと伸びた背筋。豊かな白髪を丁寧に撫でつけ、品のある立ち姿を見せている。

　柔和な面持ちには穏やかな皺が刻まれており、どこぞの温厚な老紳士という出立ちであるが、その手には刀身の長い木剣が握られていた。

「いんや、まだまだ。今はちょっと、哲学してたとこでして」

「ほう、興味深いお話です。何を考えていらしたのですか？」

「上は大火事、下は洪水……これ、なーんだってね」

　両足を振り上げ、振り下ろす動作で勢いをつけて立ち上がる。

　体の芯に重いものが残っているが、打撲の痛みなどの影響は微々たるものだ。

　軽く手足を回してそれを確認し、スバルは握ったままだった木剣をくるくると回して正面──ヴィルヘルムに突きつける。

「じゃ、またもう一手、ご指南お願いします」

「ちなみに先ほどの哲学のお答えは？」

「大した答えじゃないですよ──おねしょして逆ギレ」
"@

$pass1System = @"
You are translating a Japanese light novel into English for fluent English readers.

Produce natural, readable English prose. Do not translate literally — prioritize readability and faithfully convey the author's narrative voice.

Guidelines:
- Preserve each character's speech style as described in the glossary below.
- Translate honorifics naturally into English equivalents. Keep culturally specific honorifics only when dramatically significant. Be consistent throughout.
- When you encounter {kanji|reading} notation, use the reading to inform your translation. Do not reproduce the notation in the output.
- Preserve paragraph structure exactly: each source paragraph produces one output paragraph. Do not merge or split paragraphs.
- Preserve scene break markers exactly as they appear in the source.
- Use the glossary's English forms for all proper nouns without deviation.
- Do NOT add translator's notes, footnotes, or commentary.
- Do NOT censor or sanitize content.
- Do NOT summarize — translate the complete text.
- Output only the English translation, nothing else.
"@

# --- Inverted approach: analytical first pass, polish second pass ---

$pass1AnalyticalSystem = @"
You are a professional translator working on a Japanese light novel for fluent English readers.

First, output an <analysis> block. For each paragraph, note any translation challenges:

- **Specific terminology**: Identify specific nouns (cloud types, flora, fauna, weapons, cultural items) and verify you are using the precise English equivalent, not a generalized term.
- **Physical actions and body positions**: Parse each verb of motion carefully. These are commonly mistranslated.
- **Dialogue and voice**: Note each character's speech register. Where the source contains wordplay, puns, or riddles, brainstorm an English equivalent that preserves both meaning and humor.
- **Omission risk**: Flag any nuance that might be lost in a natural English rendering.

Then, after </analysis>, output the English translation.

Translation guidelines:
- Produce natural, readable English prose that faithfully conveys the author's narrative voice.
- Preserve paragraph structure exactly — do not merge or split paragraphs.
- Preserve scene break markers exactly as they appear in the source.
- Translate honorifics naturally. Keep culturally specific honorifics only when dramatically significant.
- When you encounter {kanji|reading} notation, use the reading to inform your translation. Do not reproduce the notation in the output.
- Do NOT add translator's notes, footnotes, or commentary in the final translation.
- Do NOT censor or sanitize content.
- Do NOT summarize — translate the complete text.
"@

$pass2PolishSystem = @"
You are a fiction editor polishing an English translation of a Japanese light novel.

The translation is already accurate. Your job is to improve readability without changing meaning:

- Smooth out any phrasing that sounds stiff, overly literal, or awkward when read aloud.
- Ensure dialogue sounds natural for each character. Casual characters should sound casual, formal characters should sound formal. Cut unnecessary hedging or filler in dialogue.
- Tighten prose: prefer active verbs, vary sentence openings, remove redundancy.
- Do NOT change proper nouns, terminology, or paragraph structure.
- Do NOT re-interpret the source — trust the translation's accuracy and only improve the English surface.

Output only the polished English translation, nothing else.
"@

function Invoke-LMStudio {
    param(
        [array]$Messages,
        [string]$Label
    )
    $body = @{
        model = $Model
        messages = $Messages
        temperature = $Temperature
        max_tokens = $MaxTokens
    } | ConvertTo-Json -Depth 5

    $start = Get-Date
    $response = Invoke-RestMethod -Uri "http://127.0.0.1:1234/v1/chat/completions" `
        -Method Post -Body $body -ContentType "application/json; charset=utf-8"
    $elapsed = (Get-Date) - $start

    $result = $response.choices[0].message.content
    $promptTok = $response.usage.prompt_tokens
    $completionTok = $response.usage.completion_tokens

    Write-Host "=== $Label ==="
    Write-Host $result
    Write-Host ""
    Write-Host "--- Stats: prompt=$promptTok completion=$completionTok time=$([math]::Round($elapsed.TotalSeconds, 1))s ---"
    Write-Host ""

    return $result
}

# --- Inverted Pass 1: analytical translation ---
$pass1Messages = @(
    @{ role = "system"; content = $pass1AnalyticalSystem }
    @{ role = "user"; content = $sourceText }
)
$pass1Result = Invoke-LMStudio -Messages $pass1Messages -Label "PASS 1 (analytical)"

# Strip the <analysis> block to get just the translation for Pass 2
$pass1Translation = ($pass1Result -split '</analysis>')[-1].Trim()
Write-Host "=== PASS 1 (translation only) ==="
Write-Host $pass1Translation
Write-Host ""

# --- Inverted Pass 2: polish only ---
$pass2Messages = @(
    @{ role = "system"; content = $pass2PolishSystem }
    @{ role = "user"; content = $pass1Translation }
)
$pass2Result = Invoke-LMStudio -Messages $pass2Messages -Label "PASS 2 (polish)"
