from robosuite.environments.base import make

# Manipulation environments
from robocasa.environments.kitchen.kitchen import Kitchen
from robocasa.environments.kitchen.multi_stage.adding_ice_to_beverages.make_ice_lemonade import (
    MakeIceLemonade,
)
from robocasa.environments.kitchen.multi_stage.adding_ice_to_beverages.make_iced_coffee import (
    MakeIcedCoffee,
)
from robocasa.environments.kitchen.multi_stage.adding_ice_to_beverages.place_equal_ice_cubes import (
    PlaceEqualIceCubes,
)
from robocasa.environments.kitchen.multi_stage.adding_ice_to_beverages.place_ice_in_cup import (
    PlaceIceInCup,
)
from robocasa.environments.kitchen.multi_stage.adding_ice_to_beverages.retrieve_ice_tray import (
    RetrieveIceTray,
)
from robocasa.environments.kitchen.multi_stage.arranging_cabinets.reset_cabinet_doors import (
    ResetCabinetDoors,
)
from robocasa.environments.kitchen.multi_stage.arranging_cabinets.gather_tableware import (
    GatherTableware,
)
from robocasa.environments.kitchen.multi_stage.arranging_cabinets.stack_cans import (
    StackCans,
)
from robocasa.environments.kitchen.multi_stage.arranging_condiments.categorize_condiments import (
    CategorizeCondiments,
)
from robocasa.environments.kitchen.multi_stage.arranging_condiments.organize_condiments import (
    OrganizeCondiments,
)
from robocasa.environments.kitchen.multi_stage.arranging_condiments.line_up_condiments import (
    LineUpCondiments,
)
from robocasa.environments.kitchen.multi_stage.arranging_buffet.arrange_buffet_dessert import (
    ArrangeBuffetDessert,
)
from robocasa.environments.kitchen.multi_stage.arranging_buffet.cut_buffet_pizza import (
    CutBuffetPizza,
)
from robocasa.environments.kitchen.multi_stage.arranging_buffet.divide_buffet_trays import (
    DivideBuffetTrays,
)
from robocasa.environments.kitchen.multi_stage.arranging_buffet.place_beverages_together import (
    PlaceBeveragesTogether,
)
from robocasa.environments.kitchen.multi_stage.arranging_buffet.tong_buffet_setup import (
    TongBuffetSetup,
)
from robocasa.environments.kitchen.multi_stage.baking.cupcake_cleanup import (
    CupcakeCleanup,
)
from robocasa.environments.kitchen.multi_stage.baking.organize_baking_ingredients import (
    OrganizeBakingIngredients,
)
from robocasa.environments.kitchen.multi_stage.baking.pastry_display import (
    PastryDisplay,
)
from robocasa.environments.kitchen.multi_stage.baking_cookies_cakes.cookie_dough_prep import (
    CookieDoughPrep,
)
from robocasa.environments.kitchen.multi_stage.baking_cookies_cakes.cool_baked_cake import (
    CoolBakedCake,
)
from robocasa.environments.kitchen.multi_stage.baking_cookies_cakes.mix_cake_frosting import (
    MixCakeFrosting,
)

from robocasa.environments.kitchen.multi_stage.boiling.cool_kettle import CoolKettle
from robocasa.environments.kitchen.multi_stage.boiling.fill_kettle import FillKettle
from robocasa.environments.kitchen.multi_stage.boiling.heat_multiple_water import (
    HeatMultipleWater,
)
from robocasa.environments.kitchen.multi_stage.boiling.microwave_teapot import (
    MicrowaveTeapot,
)
from robocasa.environments.kitchen.multi_stage.boiling.start_electric_kettle import (
    StartElectricKettle,
)
from robocasa.environments.kitchen.multi_stage.boiling.veggie_boil import VeggieBoil
from robocasa.environments.kitchen.multi_stage.brewing.arrange_tea import ArrangeTea
from robocasa.environments.kitchen.multi_stage.brewing.kettle_boiling import (
    KettleBoiling,
)
from robocasa.environments.kitchen.multi_stage.brewing.organize_coffee_condiments import (
    OrganizeCoffeeCondiments,
)
from robocasa.environments.kitchen.multi_stage.brewing.prepare_coffee import (
    PrepareCoffee,
)
from robocasa.environments.kitchen.multi_stage.broiling_fish.oven_broil_fish import (
    OvenBroilFish,
)
from robocasa.environments.kitchen.multi_stage.broiling_fish.prepare_broiling_station import (
    PrepareBroilingStation,
)
from robocasa.environments.kitchen.multi_stage.broiling_fish.remove_broiled_fish import (
    RemoveBroiledFish,
)
from robocasa.environments.kitchen.multi_stage.broiling_fish.wash_fish import (
    WashFish,
)
from robocasa.environments.kitchen.multi_stage.chopping_food.arrange_vegetables import (
    ArrangeVegetables,
    ArrangeVegetablesSimple,
)
from robocasa.environments.kitchen.multi_stage.chopping_food.bread_setup_slicing import (
    BreadSetupSlicing,
)
from robocasa.environments.kitchen.multi_stage.chopping_food.clear_cutting_board import (
    ClearCuttingBoard,
)
from robocasa.environments.kitchen.multi_stage.chopping_food.meat_transfer import (
    MeatTransfer,
)
from robocasa.environments.kitchen.multi_stage.chopping_food.organize_vegetables import (
    OrganizeVegetables,
)
from robocasa.environments.kitchen.multi_stage.chopping_vegetables.gather_cutting_tools import (
    GatherCuttingTools,
)
from robocasa.environments.kitchen.multi_stage.chopping_vegetables.cutting_tool_selection import (
    CuttingToolSelection,
)
from robocasa.environments.kitchen.multi_stage.cleaning_appliances.prep_fridge_for_cleaning import (
    PrepFridgeForCleaning,
)
from robocasa.environments.kitchen.multi_stage.cleaning_appliances.prep_sink_for_cleaning import (
    PrepSinkForCleaning,
)
from robocasa.environments.kitchen.multi_stage.cleaning_sink.clear_food_waste import (
    ClearFoodWaste,
)
from robocasa.environments.kitchen.multi_stage.cleaning_sink.clear_sink_area import (
    ClearSinkArea,
)
from robocasa.environments.kitchen.multi_stage.cleaning_sink.rinse_sink_basin import (
    RinseSinkBasin,
)
from robocasa.environments.kitchen.multi_stage.clearing_table.bowl_and_cup import (
    BowlAndCup,
)
from robocasa.environments.kitchen.multi_stage.clearing_table.candle_cleanup import (
    CandleCleanup,
)
from robocasa.environments.kitchen.multi_stage.clearing_table.clear_receptacles_for_cleaning import (
    ClearReceptaclesForCleaning,
)
from robocasa.environments.kitchen.multi_stage.clearing_table.condiment_collection import (
    CondimentCollection,
)
from robocasa.environments.kitchen.multi_stage.clearing_table.dessert_assembly import (
    DessertAssembly,
)
from robocasa.environments.kitchen.multi_stage.clearing_table.drinkware_consolidation import (
    DrinkwareConsolidation,
)
from robocasa.environments.kitchen.multi_stage.clearing_table.food_cleanup import (
    FoodCleanup,
)
from robocasa.environments.kitchen.multi_stage.defrosting_food.defrost_by_category import (
    DefrostByCategory,
)
from robocasa.environments.kitchen.multi_stage.defrosting_food.microwave_thawing import (
    MicrowaveThawing,
)
from robocasa.environments.kitchen.multi_stage.defrosting_food.microwave_thawing_fridge import (
    MicrowaveThawingFridge,
)
from robocasa.environments.kitchen.multi_stage.defrosting_food.move_to_counter import (
    MoveToCounter,
)
from robocasa.environments.kitchen.multi_stage.defrosting_food.quick_thaw import (
    QuickThaw,
)
from robocasa.environments.kitchen.multi_stage.defrosting_food.thaw_in_sink import (
    ThawInSink,
)
from robocasa.environments.kitchen.multi_stage.filling_serving_dishes.build_appetizer_plate import (
    BuildAppetizerPlate,
)
from robocasa.environments.kitchen.multi_stage.filling_serving_dishes.display_meat_variety import (
    DisplayMeatVariety,
)
from robocasa.environments.kitchen.multi_stage.filling_serving_dishes.meat_skewer_assembly import (
    MeatSkewerAssembly,
)
from robocasa.environments.kitchen.multi_stage.filling_serving_dishes.mixed_fruit_platter import (
    MixedFruitPlatter,
)
from robocasa.environments.kitchen.multi_stage.frying.assemble_cooking_array import (
    AssembleCookingArray,
)
from robocasa.environments.kitchen.multi_stage.frying.flip_sausage import (
    FlipSausage,
)
from robocasa.environments.kitchen.multi_stage.frying.frying_pan_adjustment import (
    FryingPanAdjustment,
)
from robocasa.environments.kitchen.multi_stage.frying.meal_prep_staging import (
    MealPrepStaging,
)
from robocasa.environments.kitchen.multi_stage.frying.press_chicken import (
    PressChicken,
)
from robocasa.environments.kitchen.multi_stage.frying.rocking_kebab import (
    RockingKebab,
)
from robocasa.environments.kitchen.multi_stage.frying.rotate_pan import (
    RotatePan,
)
from robocasa.environments.kitchen.multi_stage.frying.searing_meat import SearingMeat
from robocasa.environments.kitchen.multi_stage.frying.setup_frying import SetupFrying
from robocasa.environments.kitchen.multi_stage.garnishing_dishes.add_lemon_to_fish import (
    AddLemonToFish,
)
from robocasa.environments.kitchen.multi_stage.garnishing_dishes.add_sugar_cubes import (
    AddSugarCubes,
)
from robocasa.environments.kitchen.multi_stage.garnishing_dishes.garnish_cupcake import (
    GarnishCupcake,
)
from robocasa.environments.kitchen.multi_stage.garnishing_dishes.garnish_pancake import (
    GarnishPancake,
)
from robocasa.environments.kitchen.multi_stage.making_salads.prepare_cheese_station import (
    PrepareCheeseStation,
)
from robocasa.environments.kitchen.multi_stage.making_salads.wash_lettuce import (
    WashLettuce,
)
from robocasa.environments.kitchen.multi_stage.making_tea.arrange_tea_accompaniments import (
    ArrangeTeaAccompaniments,
)
from robocasa.environments.kitchen.multi_stage.making_tea.lower_heat import (
    LowerHeat,
)
from robocasa.environments.kitchen.multi_stage.making_tea.serve_tea import (
    ServeTea,
)
from robocasa.environments.kitchen.multi_stage.making_tea.stir_tea import (
    StirTea,
)
from robocasa.environments.kitchen.multi_stage.making_tea.strainer_setup import (
    StrainerSetup,
)
from robocasa.environments.kitchen.multi_stage.making_toast.bread_selection import (
    BreadSelection,
)
from robocasa.environments.kitchen.multi_stage.making_toast.cheesy_bread import (
    CheesyBread,
)
from robocasa.environments.kitchen.multi_stage.making_toast.prepare_toast import (
    PrepareToast,
)
from robocasa.environments.kitchen.multi_stage.making_toast.sweet_savory_toast_setup import (
    SweetSavoryToastSetup,
)

from robocasa.environments.kitchen.multi_stage.measuring_ingredients.choose_measuring_cup import (
    ChooseMeasuringCup,
)
from robocasa.environments.kitchen.multi_stage.measuring_ingredients.weigh_ingredients import (
    WeighIngredients,
)

from robocasa.environments.kitchen.multi_stage.meat_preparation.prep_for_tenderizing import (
    PrepForTenderizing,
)
from robocasa.environments.kitchen.multi_stage.meat_preparation.prep_marinating_meat import (
    PrepMarinatingMeat,
)
from robocasa.environments.kitchen.multi_stage.preparing_marinade.gather_marinade_ingredients import (
    GatherMarinadeIngredients,
)
from robocasa.environments.kitchen.multi_stage.preparing_marinade.place_meat_in_marinade import (
    PlaceMeatInMarinade,
)
from robocasa.environments.kitchen.multi_stage.mixing_and_blending.colorful_salsa import (
    ColorfulSalsa,
)
from robocasa.environments.kitchen.multi_stage.mixing_and_blending.setup_juicing import (
    SetupJuicing,
)
from robocasa.environments.kitchen.multi_stage.mixing_and_blending.spicy_marinade import (
    SpicyMarinade,
)
from robocasa.environments.kitchen.multi_stage.plating_food.balanced_meal_prep import (
    BalancedMealPrep,
)
from robocasa.environments.kitchen.multi_stage.plating_food.plate_store_dinner import (
    PlateStoreDinner,
)
from robocasa.environments.kitchen.multi_stage.portioning_meals.distribute_chicken import (
    DistributeChicken,
)
from robocasa.environments.kitchen.multi_stage.portioning_meals.portion_fruit_bowl import (
    PortionFruitBowl,
)
from robocasa.environments.kitchen.multi_stage.portioning_meals.portion_hot_dogs import (
    PortionHotDogs,
)
from robocasa.environments.kitchen.multi_stage.portioning_meals.portion_in_tupperware import (
    PortionInTupperware,
)
from robocasa.environments.kitchen.multi_stage.portioning_meals.portion_on_size import (
    PortionOnSize,
)
from robocasa.environments.kitchen.multi_stage.portioning_meals.portion_yogurt import (
    PortionYogurt,
)
from robocasa.environments.kitchen.multi_stage.portioning_meals.scale_portioning import (
    ScalePortioning,
)
from robocasa.environments.kitchen.multi_stage.organizing_dishes_and_containers.empty_dish_rack import (
    EmptyDishRack,
)
from robocasa.environments.kitchen.multi_stage.organizing_dishes_and_containers.organize_mugs_by_handle import (
    OrganizeMugsByHandle,
)
from robocasa.environments.kitchen.multi_stage.organizing_dishes_and_containers.stack_bowls_cabinet import (
    StackBowlsCabinet,
)
from robocasa.environments.kitchen.multi_stage.organizing_recycling.recycle_bottles_by_size import (
    RecycleBottlesBySize,
)
from robocasa.environments.kitchen.multi_stage.organizing_recycling.recycle_bottles_by_type import (
    RecycleBottlesByType,
)
from robocasa.environments.kitchen.multi_stage.organizing_recycling.recycle_soda_cans import (
    RecycleSodaCans,
)
from robocasa.environments.kitchen.multi_stage.organizing_recycling.recycle_stacked_yogurt import (
    RecycleStackedYogurt,
)
from robocasa.environments.kitchen.multi_stage.packing_lunches.pack_dessert import (
    PackDessert,
)
from robocasa.environments.kitchen.multi_stage.packing_lunches.pack_food_by_temp import (
    PackFoodByTemp,
)
from robocasa.environments.kitchen.multi_stage.packing_lunches.pack_fruit_container import (
    PackFruitContainer,
)
from robocasa.environments.kitchen.multi_stage.packing_lunches.pack_identical_lunches import (
    PackIdenticalLunches,
)
from robocasa.environments.kitchen.multi_stage.organizing_utensils.arrange_utensils_by_type import (
    ArrangeUtensilsByType,
)
from robocasa.environments.kitchen.multi_stage.organizing_utensils.organize_metalic_utensils import (
    OrganizeMetallicUtensils,
)
from robocasa.environments.kitchen.multi_stage.preparing_hot_chocolate.add_marshmallow import (
    AddMarshmallow,
)
from robocasa.environments.kitchen.multi_stage.preparing_hot_chocolate.sweeten_hot_chocolate import (
    SweetenHotChocolate,
)
from robocasa.environments.kitchen.multi_stage.preparing_sandwiches.gather_vegetables import (
    GatherVegetables,
)
from robocasa.environments.kitchen.multi_stage.preparing_sandwiches.heat_kebab_sandwich import (
    HeatKebabSandwich,
)
from robocasa.environments.kitchen.multi_stage.preparing_sandwiches.hot_dog_setup import (
    HotDogSetup,
)
from robocasa.environments.kitchen.multi_stage.preparing_sandwiches.prepare_sausage_cheese import (
    PrepareSausageCheese,
)
from robocasa.environments.kitchen.multi_stage.reheating_food.heat_mug import HeatMug
from robocasa.environments.kitchen.multi_stage.reheating_food.make_loaded_potato import (
    MakeLoadedPotato,
)
from robocasa.environments.kitchen.multi_stage.reheating_food.simmering_sauce import (
    SimmeringSauce,
)
from robocasa.environments.kitchen.multi_stage.reheating_food.waffle_reheat import (
    WaffleReheat,
)
from robocasa.environments.kitchen.multi_stage.reheating_food.warm_croissant import (
    WarmCroissant,
)
from robocasa.environments.kitchen.multi_stage.restocking_supplies.beverage_sorting import (
    BeverageSorting,
)
from robocasa.environments.kitchen.multi_stage.restocking_supplies.restock_bowls import (
    RestockBowls,
)
from robocasa.environments.kitchen.multi_stage.restocking_supplies.restock_pantry import (
    RestockPantry,
)
from robocasa.environments.kitchen.multi_stage.restocking_supplies.stocking_breakfast_foods import (
    StockingBreakfastFoods,
)
from robocasa.environments.kitchen.multi_stage.sanitize_surface.clean_microwave import (
    CleanMicrowave,
)
from robocasa.environments.kitchen.multi_stage.sanitize_surface.countertop_cleanup import (
    CountertopCleanup,
)
from robocasa.environments.kitchen.multi_stage.sanitize_surface.prep_for_sanitizing import (
    PrepForSanitizing,
)
from robocasa.environments.kitchen.multi_stage.sanitize_surface.push_utensils_to_sink import (
    PushUtensilsToSink,
)
from robocasa.environments.kitchen.multi_stage.sanitizing_cutting_board.remove_cutting_board_items import (
    RemoveCuttingBoardItems,
)
from robocasa.environments.kitchen.multi_stage.sanitizing_cutting_board.rinse_cutting_board import (
    RinseCuttingBoard,
)
from robocasa.environments.kitchen.multi_stage.sanitizing_cutting_board.sanitize_prep_cutting_board import (
    SanitizePrepCuttingBoard,
)
from robocasa.environments.kitchen.multi_stage.sanitizing_cutting_board.scrub_cutting_board import (
    ScrubCuttingBoard,
)
from robocasa.environments.kitchen.multi_stage.sauteing_vegetables.adjust_heat import (
    AdjustHeat,
)
from robocasa.environments.kitchen.multi_stage.sauteing_vegetables.butter_on_pan import (
    ButterOnPan,
)
from robocasa.environments.kitchen.multi_stage.sauteing_vegetables.place_vegetables_evenly import (
    PlaceVegetablesEvenly,
)
from robocasa.environments.kitchen.multi_stage.sauteing_vegetables.preheat_pot import (
    PreheatPot,
)
from robocasa.environments.kitchen.multi_stage.sauteing_vegetables.shake_pan import (
    ShakePan,
)
from robocasa.environments.kitchen.multi_stage.sauteing_vegetables.stir_vegetables import (
    StirVegetables,
)
from robocasa.environments.kitchen.multi_stage.sauteing_vegetables.tilt_pan import (
    TiltPan,
)
from robocasa.environments.kitchen.multi_stage.seasoning_food.lemon_seasoning_fish import (
    LemonSeasoningFish,
)
from robocasa.environments.kitchen.multi_stage.seasoning_food.seasoning_steak import (
    SeasoningSteak,
)
from robocasa.environments.kitchen.multi_stage.seasoning_food.setup_spice_station import (
    SetUpSpiceStation,
)
from robocasa.environments.kitchen.multi_stage.serving_beverages.deliver_straw import (
    DeliverStraw,
)
from robocasa.environments.kitchen.multi_stage.serving_beverages.match_cup_and_drink import (
    MatchCupAndDrink,
)
from robocasa.environments.kitchen.multi_stage.serving_beverages.prepare_cocktail_station import (
    PrepareCocktailStation,
)
from robocasa.environments.kitchen.multi_stage.serving_beverages.prepare_drink_station import (
    PrepareDrinkStation,
)
from robocasa.environments.kitchen.multi_stage.serving_beverages.setup_soda_bowl import (
    SetupSodaBowl,
)
from robocasa.environments.kitchen.multi_stage.serving_food.dessert_upgrade import (
    DessertUpgrade,
)
from robocasa.environments.kitchen.multi_stage.serving_food.pan_transfer import (
    PanTransfer,
)
from robocasa.environments.kitchen.multi_stage.serving_food.place_food_in_bowls import (
    PlaceFoodInBowls,
)
from robocasa.environments.kitchen.multi_stage.serving_food.prepare_soup_serving import (
    PrepareSoupServing,
)
from robocasa.environments.kitchen.multi_stage.serving_food.serve_steak import (
    ServeSteak,
)
from robocasa.environments.kitchen.multi_stage.serving_food.alcohol_serving_prep import (
    AlcoholServingPrep,
)
from robocasa.environments.kitchen.multi_stage.setting_the_table.align_silverware import (
    AlignSilverware,
)
from robocasa.environments.kitchen.multi_stage.setting_the_table.arrange_bread_basket import (
    ArrangeBreadBasket,
)
from robocasa.environments.kitchen.multi_stage.setting_the_table.beverage_organization import (
    BeverageOrganization,
)
from robocasa.environments.kitchen.multi_stage.setting_the_table.date_night import (
    DateNight,
)
from robocasa.environments.kitchen.multi_stage.setting_the_table.setup_bowls import (
    SetupBowls,
)
from robocasa.environments.kitchen.multi_stage.setting_the_table.setup_butter_plate import (
    SetupButterPlate,
)
from robocasa.environments.kitchen.multi_stage.setting_the_table.setup_fruit_bowl import (
    SetupFruitBowl,
)
from robocasa.environments.kitchen.multi_stage.setting_the_table.setup_wine_glasses import (
    SetupWineGlasses,
)
from robocasa.environments.kitchen.multi_stage.setting_the_table.seasoning_spice_setup import (
    SeasoningSpiceSetup,
)
from robocasa.environments.kitchen.multi_stage.setting_the_table.set_bowls_for_soup import (
    SetBowlsForSoup,
)
from robocasa.environments.kitchen.multi_stage.setting_the_table.size_sorting import (
    SizeSorting,
)
from robocasa.environments.kitchen.multi_stage.slicing_meat.retrieve_meat import (
    RetrieveMeat,
)
from robocasa.environments.kitchen.multi_stage.slicing_meat.clean_board import (
    CleanBoard,
)
from robocasa.environments.kitchen.multi_stage.slicing_meat.set_up_cutting_station import (
    SetUpCuttingStation,
)
from robocasa.environments.kitchen.multi_stage.simmering_sauces.turn_off_simmered_sauce_heat import (
    TurnOffSimmeredSauceHeat,
)
from robocasa.environments.kitchen.multi_stage.snack_preparation.bread_and_cheese import (
    BreadAndCheese,
)
from robocasa.environments.kitchen.multi_stage.snack_preparation.cereal_and_bowl import (
    CerealAndBowl,
)
from robocasa.environments.kitchen.multi_stage.snack_preparation.make_fruit_bowl import (
    MakeFruitBowl,
)
from robocasa.environments.kitchen.multi_stage.snack_preparation.veggie_dip_prep import (
    VeggieDipPrep,
)
from robocasa.environments.kitchen.multi_stage.snack_preparation.yogurt_delight_prep import (
    YogurtDelightPrep,
)
from robocasa.environments.kitchen.multi_stage.steaming_food.multistep_steaming import (
    MultistepSteaming,
)
from robocasa.environments.kitchen.multi_stage.steaming_food.steam_in_microwave import (
    SteamInMicrowave,
)
from robocasa.environments.kitchen.multi_stage.steaming_food.steam_vegetables import (
    SteamVegetables,
)
from robocasa.environments.kitchen.multi_stage.storing_leftovers.freeze_cooked_food import (
    FreezeCookedFood,
)
from robocasa.environments.kitchen.multi_stage.storing_leftovers.store_dumplings import (
    StoreDumplings,
)
from robocasa.environments.kitchen.multi_stage.storing_leftovers.store_leftovers_by_type import (
    StoreLeftoversByType,
)
from robocasa.environments.kitchen.multi_stage.storing_leftovers.store_leftovers_in_bowl import (
    StoreLeftoversInBowl,
)
from robocasa.environments.kitchen.multi_stage.tidying_cabinets_and_drawers.drawer_utensil_sort import (
    DrawerUtensilSort,
)
from robocasa.environments.kitchen.multi_stage.tidying_cabinets_and_drawers.organize_cleaning_supplies import (
    OrganizeCleaningSupplies,
)
from robocasa.environments.kitchen.multi_stage.tidying_cabinets_and_drawers.pantry_mishap import (
    PantryMishap,
)
from robocasa.environments.kitchen.multi_stage.tidying_cabinets_and_drawers.shaker_shuffle import (
    ShakerShuffle,
)
from robocasa.environments.kitchen.multi_stage.tidying_cabinets_and_drawers.snack_sorting import (
    SnackSorting,
)
from robocasa.environments.kitchen.multi_stage.toasting_bread.get_toasted_bread import (
    GetToastedBread,
)
from robocasa.environments.kitchen.multi_stage.toasting_bread.serve_warm_croissant import (
    ServeWarmCroissant,
)
from robocasa.environments.kitchen.multi_stage.toasting_bread.toast_bagel import (
    ToastBagel,
)
from robocasa.environments.kitchen.multi_stage.toasting_bread.toast_baguette import (
    ToastBaguette,
)
from robocasa.environments.kitchen.multi_stage.toasting_bread.toast_on_correct_rack import (
    ToastOnCorrectRack,
)
from robocasa.environments.kitchen.multi_stage.washing_dishes.clear_sink import (
    ClearSink,
)
from robocasa.environments.kitchen.multi_stage.washing_dishes.collect_washing_supplies import (
    CollectWashingSupplies,
)
from robocasa.environments.kitchen.multi_stage.washing_dishes.dry_dishes import (
    DryDishes,
)
from robocasa.environments.kitchen.multi_stage.washing_dishes.dry_drinkware import (
    DryDrinkware,
)
from robocasa.environments.kitchen.multi_stage.washing_dishes.load_dishwasher import (
    LoadDishwasher,
)
from robocasa.environments.kitchen.multi_stage.washing_dishes.prepare_dishwasher import (
    PrepareDishwasher,
)
from robocasa.environments.kitchen.multi_stage.washing_dishes.place_dishes_by_sink import (
    PlaceDishesBySink,
)
from robocasa.environments.kitchen.multi_stage.washing_dishes.divide_basins import (
    DivideBasins,
)
from robocasa.environments.kitchen.multi_stage.washing_dishes.pre_rinse_station import (
    PreRinseStation,
)
from robocasa.environments.kitchen.multi_stage.washing_dishes.pre_soak_pan import (
    PreSoakPan,
)
from robocasa.environments.kitchen.multi_stage.washing_dishes.return_washing_supplies import (
    ReturnWashingSupplies,
)
from robocasa.environments.kitchen.multi_stage.washing_dishes.rinse_bowls import (
    RinseBowls,
)
from robocasa.environments.kitchen.multi_stage.washing_dishes.rinse_fragile_item import (
    RinseFragileItem,
)
from robocasa.environments.kitchen.multi_stage.washing_dishes.scrub_bowl import (
    ScrubBowl,
)
from robocasa.environments.kitchen.multi_stage.washing_dishes.soak_sponge import (
    SoakSponge,
)
from robocasa.environments.kitchen.multi_stage.washing_dishes.sorting_cleanup import (
    SortingCleanup,
)
from robocasa.environments.kitchen.multi_stage.washing_dishes.stack_bowls import (
    StackBowlsInSink,
)
from robocasa.environments.kitchen.multi_stage.washing_dishes.transport_cookware import (
    TransportCookware,
)
from robocasa.environments.kitchen.multi_stage.washing_fruits_and_vegetables.afterwash_sorting import (
    AfterwashSorting,
)
from robocasa.environments.kitchen.multi_stage.washing_fruits_and_vegetables.clear_clutter import (
    ClearClutter,
)
from robocasa.environments.kitchen.multi_stage.washing_fruits_and_vegetables.drain_veggies import (
    DrainVeggies,
)
from robocasa.environments.kitchen.multi_stage.washing_fruits_and_vegetables.prewash_food_assembly import (
    PrewashFoodAssembly,
)
from robocasa.environments.kitchen.multi_stage.washing_fruits_and_vegetables.airdry_fruit import (
    AirDryFruit,
)
from robocasa.environments.kitchen.multi_stage.washing_fruits_and_vegetables.gather_produce_washing import (
    GatherProduceWashing,
)

from robocasa.environments.kitchen.multi_stage.washing_fruits_and_vegetables.wash_fruit_colander import (
    WashFruitColander,
)
from robocasa.environments.kitchen.multi_stage.washing_fruits_and_vegetables.prepare_vegetable_roasting import (
    PrepareVegetableRoasting,
)
from robocasa.environments.kitchen.multi_stage.washing_fruits_and_vegetables.utilize_water_variance import (
    UtilizeWaterVariance,
)
from robocasa.environments.kitchen.multi_stage.washing_fruits_and_vegetables.clear_sink_space import (
    ClearSinkSpace,
)
from robocasa.environments.kitchen.multi_stage.washing_fruits_and_vegetables.prewash_food_sorting import (
    PrewashFoodSorting,
)

from robocasa.environments.kitchen.single_stage.kitchen_coffee import (
    StartCoffeeMachine,
    CoffeeServeMug,
    CoffeeSetupMug,
)
from robocasa.environments.kitchen.single_stage.kitchen_doors import (
    OpenDoor,
    CloseDoor,
    OpenCabinet,
    CloseCabinet,
    OpenMicrowave,
    CloseMicrowave,
    OpenFridge,
    CloseFridge,
    OpenDishwasher,
    CloseDishwasher,
    OpenToasterOvenDoor,
    CloseToasterOvenDoor,
    OpenOven,
    CloseOven,
)
from robocasa.environments.kitchen.single_stage.kitchen_drawer import (
    OpenDrawer,
    CloseDrawer,
    SlideDishwasherRack,
)
from robocasa.environments.kitchen.single_stage.kitchen_electric_kettle import (
    CloseElectricKettleLid,
    OpenElectricKettleLid,
    TurnOnElectricKettle,
)
from robocasa.environments.kitchen.single_stage.kitchen_microwave import (
    TurnOnMicrowave,
    TurnOffMicrowave,
)
from robocasa.environments.kitchen.single_stage.kitchen_navigate import NavigateKitchen

from robocasa.environments.kitchen.single_stage.kitchen_oven import (
    PreheatOven,
    SlideOvenRack,
)

from robocasa.environments.kitchen.single_stage.kitchen_pnp import (
    PnPCounterToCabinet,
    PnPCabinetToCounter,
    PnPCounterToMicrowave,
    PnPMicrowaveToCounter,
    PnPCounterToSink,
    PnPSinkToCounter,
    PnPCounterToStove,
    PnPStoveToCounter,
    PnPCounterToOven,
    PnPOvenToCounter,
    PnPToasterToCounter,
    PnPCounterToToasterOven,
    PnPToasterOvenToCounter,
    PnPCounterToStandMixer,
)
from robocasa.environments.kitchen.single_stage.kitchen_sink import (
    TurnOnSinkFaucet,
    TurnOffSinkFaucet,
    TurnSinkSpout,
    AdjustWaterTemperature,
)
from robocasa.environments.kitchen.single_stage.kitchen_stand_mixer import (
    OpenStandMixerHead,
    CloseStandMixerHead,
)
from robocasa.environments.kitchen.single_stage.kitchen_stove import (
    TurnOnStove,
    TurnOffStove,
)
from robocasa.environments.kitchen.single_stage.kitchen_toaster_oven import (
    AdjustToasterOvenTemperature,
    SlideToasterOvenRack,
    TurnOnToasterOven,
)
from robocasa.environments.kitchen.single_stage.kitchen_toaster import (
    TurnOnToaster,
)

try:
    import mimicgen
except ImportError:
    print(
        "WARNING: mimicgen environments not imported since mimicgen is not installed!"
    )

from robocasa.environments import ALL_KITCHEN_ENVIRONMENTS

# from robosuite.controllers import ALL_CONTROLLERS, load_controller_config
from robosuite.controllers import ALL_PART_CONTROLLERS, load_composite_controller_config
from robosuite.environments import ALL_ENVIRONMENTS
from robosuite.models.grippers import ALL_GRIPPERS
from robosuite.robots import ALL_ROBOTS

import mujoco

assert (
    mujoco.__version__ == "3.3.1"
), "MuJoCo version must be 3.3.1. Please run pip install mujoco==3.3.1"

import numpy

assert numpy.__version__ in [
    "2.2.5",
], "numpy version must be 2.2.5. Please install this version."

import robosuite

robosuite_version = [int(e) for e in robosuite.__version__.split(".")]
robosuite_check = True
if robosuite_version[0] < 1:
    robosuite_check = False
if robosuite_version[0] == 1 and robosuite_version[1] < 5:
    robosuite_check = False
if robosuite_version[0] == 1 and robosuite_version[1] == 5 and robosuite_version[2] < 2:
    robosuite_check = False
assert (
    robosuite_check
), "robosuite version must be >=1.5.2 Please install the correct version"

__version__ = "0.5.1"
__logo__ = """
      ;     /        ,--.
     ["]   ["]  ,<  |__**|
    /[_]\  [~]\/    |//  |
     ] [   OOO      /o|__|
"""
