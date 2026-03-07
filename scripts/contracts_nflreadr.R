library(nflreadr)
library(dplyr)

x <- raw_data %>%
  filter(player == "Johnathan Crompton")
# 1. Standard Clean
raw_data <- load_contracts() %>%
  select(-where(is.list)) %>%
  filter(!is.na(gsis_id)) %>%
  mutate(year = as.numeric(year_signed)) %>%
  filter(year >= 2010)

# 2. Career Analysis
career_analysis <- raw_data %>%
  group_by(gsis_id) %>%
  arrange(year_signed, .by_group = TRUE) %>%
  mutate(contract_order = row_number()) %>%
  mutate(
    # TAG LOGIC: 1 year, fully guaranteed (use near-equality for float safety),
    # and high value. Floor of 10M covers all positions across all recent years.
    is_tag = if_else(
      contract_order > 1 &
        years == 1 &
        #abs(value - guaranteed) < 0.01 &   # float-safe equality check
        value >= 10.0,
      TRUE, FALSE
    ),
    
    # REAL CONTRACT LOGIC:
    # Multi-year: value > 2M AND years >= 2
    # Single-year: value >= 10M (tags or big 1-year deals)
    is_real_contract = if_else(
      contract_order > 1 &
        is_tag == FALSE &
        ((value > 2.0 & years >= 2) | value >= 10.0),
      TRUE, FALSE
    )
  ) %>%
  ungroup()

# 3. Player Summary — one row per player
final_summary <- career_analysis %>%
  group_by(gsis_id, player, draft_year) %>%
  summarise(
    first_contract_year    = min(year_signed),
    got_real_2nd_contract  = any(is_real_contract),
    got_tagged             = any(is_tag),
    # Binary target: made it if they got a real 2nd deal OR were tagged
    made_it                = got_real_2nd_contract | got_tagged,
    .groups = "drop"
  )

# Verify key players
filter(final_summary, player %in% c("Kyle Pitts", "Tee Higgins", "Kyle Peko"))

write.csv(final_summary, "C:\\Users\\sffra\\Downloads\\BSE 2025-2026\\nfl-draft-nlp\\data\\raw\\contract\\nfl_contracts.csv")
